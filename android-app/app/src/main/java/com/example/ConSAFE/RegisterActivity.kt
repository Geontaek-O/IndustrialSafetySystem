package com.example.ConSAFE

import android.content.Intent
import android.content.pm.ActivityInfo
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.WindowManager
import android.widget.*
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File

/**
 * [RegisterActivity]
 * 신규 작업자 회원가입 및 초기 안면 데이터(영상) 수집 처리 화면.
 */
class RegisterActivity : AppCompatActivity() {

    // 싱글톤 API 인터페이스 인스턴스
    private val api = ApiService.create()

    // 촬영 완료된 안면 데이터 영상 파일
    private var recordedVideoFile: File? = null

    // 업로드 진행 상태 플래그 (생명주기 이탈/회전 시 코루틴 취소 방어용)
    private var isUploading = false
    private lateinit var backCallback: OnBackPressedCallback

    // UI 컴포넌트 선언
    private lateinit var etEmail: EditText
    private lateinit var etPw: EditText
    private lateinit var etName: EditText
    private lateinit var rgGender: RadioGroup
    private lateinit var etPhone: EditText
    private lateinit var etBirthdate: EditText
    private lateinit var btnFace: Button
    private lateinit var tvFaceStatus: TextView
    private lateinit var btnSubmit: Button

    /* --- 안면 인식 영상 촬영 결과 수신 콜백 (단일 MP4 파일) --- */
    private val faceGuideLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK) {
            val path = result.data?.getStringExtra(FaceGuideActivity.EXTRA_VIDEO_PATH)
            if (!path.isNullOrEmpty()) {
                recordedVideoFile = File(path)
                tvFaceStatus.text = "✅ 얼굴 촬영 완료"
                tvFaceStatus.setTextColor(getColor(android.R.color.holo_green_dark))
            } else {
                tvFaceStatus.text = "❌ 파일 저장 실패, 다시 시도해주세요."
                tvFaceStatus.setTextColor(getColor(android.R.color.holo_red_dark))
            }
        } else {
            tvFaceStatus.text = "❌ 촬영이 취소되었습니다."
            tvFaceStatus.setTextColor(getColor(android.R.color.holo_red_dark))
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_register)

        // UI 컴포넌트 바인딩
        etEmail     = findViewById(R.id.etRegEmail)
        etPw        = findViewById(R.id.etRegPw)
        etName      = findViewById(R.id.etRegName)
        etPhone     = findViewById(R.id.etRegPhone)
        etBirthdate = findViewById(R.id.etRegBirthdate)
        rgGender    = findViewById(R.id.rgGender)
        btnFace      = findViewById(R.id.btnFaceCapture)
        tvFaceStatus = findViewById(R.id.tvFaceStatus)
        btnSubmit    = findViewById(R.id.btnRegisterSubmit)

        // 카메라 촬영 화면 이동
        btnFace.setOnClickListener {
            faceGuideLauncher.launch(
                Intent(this, FaceGuideActivity::class.java)
            )
        }

        // 업로드 중 뒤로가기(백스와이프) 차단 콜백
        // - Activity 파괴로 인한 비동기 업로드 취소 방지
        backCallback = object : OnBackPressedCallback(false) {
            override fun handleOnBackPressed() {
                Toast.makeText(
                    this@RegisterActivity,
                    "등록을 처리 중입니다. 잠시만 기다려주세요.",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }
        onBackPressedDispatcher.addCallback(this, backCallback)

        // 가입 실행
        btnSubmit.setOnClickListener { validateAndSubmit() }

        // 전화번호 입력 텍스트 포맷팅 (010-1234-5678 자동 변환)
        etPhone.hint = "010-1234-5678"
        etPhone.addTextChangedListener(object : TextWatcher {
            private var isFormatting = false
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            override fun afterTextChanged(s: Editable?) {
                if (isFormatting || s == null) return
                isFormatting = true
                val digits = s.toString().replace("-", "")
                val formatted = StringBuilder()
                for (i in digits.indices) {
                    if (i == 3 || i == 7) formatted.append("-")
                    formatted.append(digits[i])
                }
                s.replace(0, s.length, formatted.toString())
                isFormatting = false
            }
        })

        // 생년월일 입력 텍스트 포맷팅 (YYYY-MM-DD 자동 변환)
        etBirthdate.hint = "2000-01-01"
        etBirthdate.addTextChangedListener(object : TextWatcher {
            private var isFormatting = false
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            override fun afterTextChanged(s: Editable?) {
                if (isFormatting || s == null) return
                isFormatting = true
                val digits = s.toString().replace("-", "")
                val formatted = StringBuilder()
                for (i in digits.indices) {
                    if (i == 4 || i == 6) formatted.append("-")
                    formatted.append(digits[i])
                }
                s.replace(0, s.length, formatted.toString())
                isFormatting = false
            }
        })
    }

    /* --- 입력 폼 유효성 검증 --- */
    private fun validateAndSubmit() {
        val email     = etEmail.text.toString().trim()
        val pw        = etPw.text.toString()
        val name      = etName.text.toString().trim()
        val phone     = etPhone.text.toString().trim()
        val birthdate = etBirthdate.text.toString().trim()
        val gender    = when (rgGender.checkedRadioButtonId) {
            R.id.rbMale   -> "남성"
            R.id.rbFemale -> "여성"
            else          -> ""
        }
        val videoFile = recordedVideoFile

        // 필수 필드 입력 확인 및 제약 조건 검사
        when {
            email.isEmpty()   -> { etEmail.error = "이메일을 입력하세요"; return }
            pw.length < 6     -> { etPw.error = "비밀번호는 6자 이상"; return }
            name.isEmpty()    -> { etName.error = "성명을 입력하세요"; return }
            phone.isEmpty()   -> { etPhone.error = "전화번호를 입력하세요"; return }
            birthdate.isEmpty() -> { etBirthdate.error = "생년월일을 입력하세요"; return }
            gender.isEmpty()  -> { Toast.makeText(this, "성별을 선택하세요.", Toast.LENGTH_SHORT).show(); return }
            videoFile == null -> { Toast.makeText(this, "얼굴을 촬영해주세요.", Toast.LENGTH_SHORT).show(); return }
        }

        submitRegistration(email, pw, name, phone, birthdate, gender, videoFile!!)
    }

    /* --- 데이터 및 파일 업로드 처리 --- */
    private fun submitRegistration(
        email: String, pw: String, name: String,
        phone: String, birthdate: String, gender: String, videoFile: File
    ) {
        // UI 잠금 및 생명주기 보호 로직 가동
        btnSubmit.isEnabled = false
        btnSubmit.text = "등록 중..."
        beginUploadGuard()

        // Multipart 전송용 데이터 파싱
        val textType = "text/plain".toMediaType()
        val videoPart = MultipartBody.Part.createFormData(
            "video", videoFile.name,
            videoFile.asRequestBody("video/mp4".toMediaType())
        )

        lifecycleScope.launch {
            try {
                // ApiService 연동
                val response = api.register(
                    name      = name.toRequestBody(textType),
                    gender    = gender.toRequestBody(textType),
                    email     = email.toRequestBody(textType),
                    password  = pw.toRequestBody(textType),
                    phone     = phone.toRequestBody(textType),
                    birthdate = birthdate.toRequestBody(textType),
                    video     = videoPart
                )

                // 결과 처리
                if (response.isSuccessful && response.body()?.success == true) {
                    Toast.makeText(
                        this@RegisterActivity,
                        response.body()!!.message,
                        Toast.LENGTH_LONG
                    ).show()
                    finish() // 완료 시 로그인 화면으로 회귀
                } else {
                    val msg = response.errorBody()?.string() ?: "등록 실패"
                    Toast.makeText(this@RegisterActivity, msg, Toast.LENGTH_LONG).show()
                }
            } catch (e: Exception) {
                Toast.makeText(this@RegisterActivity, "서버 연결 실패: ${e.message}", Toast.LENGTH_SHORT).show()
            } finally {
                // UI 잠금 해제 및 생명주기 보호 해제
                endUploadGuard()
                btnSubmit.isEnabled = true
                btnSubmit.text = "회원가입"
            }
        }
    }

    /* ================================================================
     * 대용량 파일 업로드 세션 보호 로직
     * - 화면 꺼짐(Wakelock), 화면 회전, 뒤로가기 동작을 강제 제한.
     * - 영상 업로드 중 Activity 파괴 시 발생하는 lifecycleScope 취소 현상 방지.
     * ================================================================ */
    private fun beginUploadGuard() {
        isUploading = true
        backCallback.isEnabled = true
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_LOCKED
    }

    private fun endUploadGuard() {
        isUploading = false
        backCallback.isEnabled = false
        window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED
    }
}