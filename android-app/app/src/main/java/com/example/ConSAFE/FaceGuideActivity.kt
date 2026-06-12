package com.example.ConSAFE

import android.content.Intent
import android.os.Bundle
import android.os.CountDownTimer
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.*
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.io.File

/**
 * 신규 작업자의 안면 인식 데이터(영상)를 수집하기 위한 가이드 촬영 액티비티.
 */
class FaceGuideActivity : AppCompatActivity() {

    /* --- UI 구성요소 --- */
    private lateinit var previewView: PreviewView
    private lateinit var tvStepTitle: TextView
    private lateinit var tvStepGuide: TextView
    private lateinit var tvCountdown: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var btnAction:   Button
    private lateinit var btnCancel:   Button

    /* --- CameraX 관련 객체 --- */
    private var cameraProvider:  ProcessCameraProvider? = null
    private var videoCapture:    VideoCapture<Recorder>? = null
    private var activeRecording: Recording? = null  // 현재 진행 중인 녹화 세션 제어 객체

    /* --- 촬영 단계 정의 (정면 → 우측 → 좌측) --- */
    private data class Step(val title: String, val guide: String, val icon: String)

    private val steps = listOf(
        Step("1단계 / 3단계", "정면을 바라봐 주세요",   "😐"),
        Step("2단계 / 3단계", "오른쪽을 바라봐 주세요", "👉"),
        Step("3단계 / 3단계", "왼쪽을 바라봐 주세요",   "👈")
    )

    private var currentStep = 0
    private var outputFile: File? = null

    /* --- 녹화 상태 관리 플래그 ---
     * 비동기로 동작하는 CameraX 콜백과 UI 이벤트 간의 충돌을 막기 위한 핵심 상태값들입니다.
     */
    private var recordingStarted = false  // 전체 녹화 세션이 최초 1회 시작되었는지 여부
    private var isRecording      = false  // 현재 물리적으로 영상이 기록되고 있는 중인지 (pause 상태가 아님)
    private var stepDone         = false  // 특정 단계(5초) 촬영이 완료되어 pause() 대기 중인 상태
    private var finishing        = false  // 마지막 3단계가 끝나고 stop() 및 파일 저장 처리가 진행 중인 상태
    private var userCancelled    = false  // 사용자가 명시적으로 '취소' 버튼을 누른 상태
    private var errorHandled     = false  // 에러 콜백 중복 실행을 막기 위한 방어 플래그

    private var countdownTimer: CountDownTimer? = null

    companion object {
        const val EXTRA_VIDEO_PATH = "video_path"
        const val TAG              = "FaceGuide"
        const val STEP_SECONDS     = 5  // 각 단계별 촬영 시간(초)
    }

    /* --- 필수 권한 요청 (오디오 없이 카메라 권한만 요구) --- */
    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera()
        else {
            Toast.makeText(this, "카메라 권한이 필요합니다.", Toast.LENGTH_SHORT).show()
            setResult(RESULT_CANCELED); finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_face_guide)

        previewView = findViewById(R.id.facePreviewView)
        tvStepTitle = findViewById(R.id.tvStepTitle)
        tvStepGuide = findViewById(R.id.tvStepGuide)
        tvCountdown = findViewById(R.id.tvFaceCountdown)
        progressBar = findViewById(R.id.stepProgressBar)
        btnAction   = findViewById(R.id.btnAction)
        btnCancel   = findViewById(R.id.btnCancel)

        btnAction.setOnClickListener { onActionClicked() }
        btnCancel.setOnClickListener { cancelAndFinish() }

        // 녹화 중 뒤로가기 버튼(제스처)에 의한 비정상 종료 방지
        onBackPressedDispatcher.addCallback(this,
            object : OnBackPressedCallback(true) {
                override fun handleOnBackPressed() {
                    // 녹화 중이거나 저장 중이 아닐 때만 취소 로직 수행
                    if (!isRecording && !finishing) {
                        isEnabled = false
                        cancelAndFinish()
                    }
                }
            })

        permissionLauncher.launch(android.Manifest.permission.CAMERA)
    }

    /* ================================================================
     * 버튼 클릭 상태 분기 처리
     * ================================================================ */
    private fun onActionClicked() {
        when {
            // 1. 녹화 세션 최초 시작 (1단계 진입)
            !recordingStarted -> startSingleRecording()

            // 2. 현재 단계 완료(pause 상태) + 다음 단계가 남아있을 경우 녹화 재개 (2, 3단계 진입)
            stepDone && currentStep < steps.size - 1 -> {
                currentStep++
                stepDone = false
                updateStepUI()
                resumeForNextStep()
            }
            // 3. 마지막 단계 완료 후에는 자동 종료 시퀀스(finishing)를 타므로 무시함
        }
    }

    /* ================================================================
     * CameraX 초기화 및 바인딩
     * ================================================================ */
    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            cameraProvider = future.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            // 서버 업로드 용량 및 화질을 고려한 해상도 전략 (HD -> FHD -> 최고화질 순)
            val qualitySelector = QualitySelector.fromOrderedList(
                listOf(Quality.HD, Quality.FHD, Quality.HIGHEST),
                FallbackStrategy.higherQualityOrLowerThan(Quality.SD)
            )
            videoCapture = VideoCapture.withOutput(
                Recorder.Builder().setQualitySelector(qualitySelector).build()
            )

            try {
                cameraProvider?.unbindAll()
                cameraProvider?.bindToLifecycle(
                    this, CameraSelector.DEFAULT_FRONT_CAMERA, preview, videoCapture
                )
                updateStepUI()
                btnAction.isEnabled = true
            } catch (e: Exception) {
                Log.e(TAG, "카메라 바인딩 실패: ${e.message}")
                tvStepGuide.text = "카메라 초기화 실패"
            }
        }, ContextCompat.getMainExecutor(this))
    }

    /* ================================================================
     * UI 업데이트 (현재 단계에 맞춰 텍스트 및 프로그레스바 갱신)
     * ================================================================ */
    private fun updateStepUI() {
        val step = steps[currentStep]
        tvStepTitle.text     = step.title
        tvStepGuide.text     = "${step.icon}  ${step.guide}"
        progressBar.progress = currentStep * 100 / steps.size
        btnAction.text       = if (currentStep == 0) "촬영 시작" else "촬영 중..."
        tvCountdown.visibility = View.GONE
    }

    /* ================================================================
     * 단일 녹화 세션 시작 (최초 1회만 호출)
     * - 이 시점부터 MP4 파일 쓰기가 시작되며, 녹화 세션은 Activity 종료 시까지 유지됩니다.
     * - 각 단계의 전환은 새로운 세션을 파는 것이 아니라 pause() / resume() 콜백에 의존합니다.
     * ================================================================ */
    private fun startSingleRecording() {
        val vc = videoCapture ?: return
        // 파일 덮어쓰기 충돌 방지를 위해 타임스탬프 기반 고유 파일명 생성
        val file = File(cacheDir, "face_${System.currentTimeMillis()}.mp4")
        outputFile = file
        val opts = FileOutputOptions.Builder(file).build()

        recordingStarted = true
        btnAction.isEnabled = false
        btnCancel.isEnabled = false

        // 안면 인식용이므로 오디오(RECORD_AUDIO 권한)는 제외하고 영상만 기록
        activeRecording = vc.output
            .prepareRecording(this, opts)
            .start(ContextCompat.getMainExecutor(this)) { event ->
                // CameraX의 비동기 상태 이벤트 콜백 (메인 스레드에서 실행 보장)
                when (event) {
                    is VideoRecordEvent.Start -> {
                        Log.d(TAG, "녹화 시작")
                        if (isFinishing || isDestroyed) return@start
                        beginStepCountdown()              // 1단계 카운트다운 시작
                    }
                    is VideoRecordEvent.Pause -> {
                        Log.d(TAG, "일시정지 — 단계 ${currentStep + 1} 완료")
                        if (isFinishing || isDestroyed) return@start
                        onStepPaused()                    // 다음 단계를 위한 UI 대기 상태 전환
                    }
                    is VideoRecordEvent.Resume -> {
                        Log.d(TAG, "재개 — 단계 ${currentStep + 1}")
                        if (isFinishing || isDestroyed) return@start
                        beginStepCountdown()              // 2, 3단계 카운트다운 시작
                    }
                    is VideoRecordEvent.Finalize -> {
                        Log.d(TAG, "Finalize — 오류=${event.hasError()}, 크기=${file.length()}")

                        // 사용자가 명시적 취소한 경우 찌꺼기 파일 완전 삭제
                        if (userCancelled) {
                            runCatching { file.delete() }
                            return@start
                        }

                        // 생명주기 종료 중이면 뷰 참조 크래시 방지를 위해 콜백 무시
                        if (isFinishing || isDestroyed) return@start

                        // 정상 종료 및 파일이 비어있지 않은 경우에만 결과 반환
                        if (!event.hasError() && file.length() > 0) {
                            returnResult(file)
                        } else {
                            onRecordingError()
                        }
                    }
                    else -> { /* Status(진행률 등) 이벤트는 사용하지 않으므로 무시 */ }
                }
            }
    }

    /* ================================================================
     * 녹화 재개 (2, 3단계 진입 시 호출)
     * ================================================================ */
    private fun resumeForNextStep() {
        btnAction.isEnabled = false
        btnCancel.isEnabled = false
        try {
            activeRecording?.resume()
        } catch (e: Exception) {
            Log.e(TAG, "resume 실패: ${e.message}")
            onRecordingError()
        }
    }

    /* ================================================================
     * 단계별 카운트다운 타이머 (STEP_SECONDS 초 단위)
     * - 지정된 시간이 끝나면 마지막 단계인지 여부에 따라 pause() 또는 stop()을 트리거합니다.
     * ================================================================ */
    private fun beginStepCountdown() {
        isRecording = true
        stepDone    = false
        btnAction.text = "촬영 중..."
        tvCountdown.visibility = View.VISIBLE
        tvCountdown.text = STEP_SECONDS.toString()

        countdownTimer?.cancel()
        countdownTimer = object : CountDownTimer((STEP_SECONDS * 1000).toLong(), 1000) {
            override fun onTick(ms: Long) {
                tvCountdown.text = ((ms / 1000) + 1).toString()
            }
            override fun onFinish() {
                // 비동기 상태 전환 중 중복 클릭을 막기 위해 UI 즉시 차단
                btnAction.isEnabled = false
                btnCancel.isEnabled = false

                if (currentStep < steps.size - 1) {
                    // 1, 2단계: 녹화를 잠시 멈추고(Pause) 사용자 대기 유도
                    tvCountdown.text = "처리 중..."
                    try {
                        activeRecording?.pause()
                    } catch (e: Exception) {
                        Log.e(TAG, "pause 실패: ${e.message}")
                        onRecordingError()
                    }
                } else {
                    // 3단계: 최종 녹화 세션 종료(Stop) -> Finalize 이벤트 호출됨
                    finishing = true
                    isRecording = false
                    tvCountdown.text     = "저장 중..."
                    tvStepGuide.text     = "✅ 모든 단계 완료! 영상을 저장하는 중..."
                    progressBar.progress = 100
                    try {
                        activeRecording?.stop()
                    } catch (e: Exception) {
                        Log.e(TAG, "stop 실패: ${e.message}")
                        onRecordingError()
                    }
                }
            }
        }.start()
    }

    /* ================================================================
     * 일시정지 완료 후 처리 (VideoRecordEvent.Pause 수신 시 호출)
     * ================================================================ */
    private fun onStepPaused() {
        isRecording = false
        stepDone    = true
        btnCancel.isEnabled  = true
        progressBar.progress = (currentStep + 1) * 100 / steps.size
        tvCountdown.visibility = View.GONE

        // 다음 단계를 위한 가이드 텍스트 및 버튼 활성화
        val next = steps[currentStep + 1]
        tvStepGuide.text    = "✅ 완료! 다음: ${next.icon} ${next.guide}"
        btnAction.text      = "다음 단계"
        btnAction.isEnabled = true
    }

    /* ================================================================
     * 촬영 성공 시 결과 데이터(파일 경로) 반환
     * ================================================================ */
    private fun returnResult(file: File) {
        countdownTimer?.cancel()
        setResult(RESULT_OK, Intent().apply {
            putExtra(EXTRA_VIDEO_PATH, file.absolutePath)
        })
        finish()
    }

    /* ================================================================
     * 녹화 오류 발생 시 복구 로직
     * - 오류가 발생하면 꼬인 세션을 완전히 날리고 처음(1단계)부터 다시 시작하도록 상태를 초기화합니다.
     * ================================================================ */
    private fun onRecordingError() {
        // 비동기 콜백 체인에서 여러 번 불리는 것을 방지 (1회 보장)
        if (errorHandled || userCancelled) return
        errorHandled = true

        countdownTimer?.cancel()
        if (!isFinishing && !isDestroyed) {
            Toast.makeText(this, "녹화 오류가 발생했습니다. 처음부터 다시 시도해주세요.", Toast.LENGTH_LONG).show()
        }

        // 진행 중이던 세션 종료 및 비정상 파일 삭제
        try { activeRecording?.stop() } catch (_: Exception) {}
        outputFile?.delete()

        // 모든 플래그 초기화
        activeRecording  = null
        outputFile       = null
        recordingStarted = false
        isRecording      = false
        stepDone         = false
        finishing        = false
        currentStep      = 0
        errorHandled     = false

        if (isFinishing || isDestroyed) return
        updateStepUI()
        btnAction.isEnabled = true
        btnCancel.isEnabled = true
    }

    /* ================================================================
     * 촬영 취소 및 액티비티 종료
     * ================================================================ */
    private fun cancelAndFinish() {
        userCancelled = true
        countdownTimer?.cancel()
        try { activeRecording?.stop() } catch (_: Exception) {}
        activeRecording = null
        outputFile?.delete()
        setResult(RESULT_CANCELED)
        finish()
    }

    /* ================================================================
     * 생명주기 해제 시 리소스 완전 정리
     * ================================================================ */
    override fun onDestroy() {
        super.onDestroy()
        countdownTimer?.cancel()
        try { activeRecording?.stop() } catch (_: Exception) {}
        cameraProvider?.unbindAll()
    }
}