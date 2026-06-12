package com.example.ConSAFE

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.res.ColorStateList
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.os.VibrationEffect
import android.os.Vibrator
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeoutOrNull
import kotlin.math.pow
import androidx.core.app.NotificationCompat
import android.app.NotificationChannel
import android.app.NotificationManager
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import android.net.Uri
import android.os.PowerManager
import android.provider.Settings
import android.speech.tts.TextToSpeech
import java.util.Locale

/**
 * [MainActivity]
 * 앱의 메인 대시보드 화면.
 * - 주요 기능: 모니터링 서비스 제어, 위치/센서/심박수 상태 표시, 위험구역 및 긴급 알림 수신.
 */
class MainActivity : AppCompatActivity() {

    // API 연동 인스턴스
    private val api = ApiService.create()

    // UI 컴포넌트
    private lateinit var tvSteps:       TextView
    private lateinit var tvUserName:    TextView
    private lateinit var tvCurrentZone: TextView
    private lateinit var ivProfile:     android.widget.ImageView
    private lateinit var tvTemperature: TextView
    private lateinit var tvHumidity:    TextView
    private lateinit var tvHeatAlert:   TextView
    private lateinit var btnStart:      Button
    private lateinit var btnStop:       Button
    private lateinit var btnBleConnect: Button

    // 모니터링 서비스 연동 상태 관리
    private var monitoringService: MonitoringService? = null
    private var isBound      = false
    private var isMonitoring = false

    // IoT 센서 데이터 전송 필터링용 이전 상태값
    private var lastTemp  = 0f
    private var lastHumid = 0f

    // 디버그 및 시연용 패널
    private var debugDialog: DebugPanelDialog? = null

    // 위치 데이터 전송 제어 (주기 및 거리 필터링)
    private var lastSentLat   = 0.0
    private var lastSentLon   = 0.0
    private var lastSentTime  = 0L
    private val SEND_INTERVAL_MS   = 5000L
    private val MIN_DISTANCE_METER = 3.0

    // 심박수 측정 매니저 및 UI
    private lateinit var tvHeartRate:   TextView
    private lateinit var tvHrAlert:     TextView
    private lateinit var btnHrRefresh:  Button
    private val heartRateManager by lazy { HeartRateManager(this) }

    // 음성 경고용 TTS
    private var tts: TextToSpeech? = null
    private var isTtsReady = false

    // Health Connect 권한 요청 런처
    private val hrPermissionLauncher = registerForActivityResult(
        androidx.health.connect.client.PermissionController
            .createRequestPermissionResultContract()
    ) { granted ->
        Log.d("HeartRate", "Health Connect 권한 결과: $granted")
        if (granted.containsAll(heartRateManager.permissions)) {
            Log.d("HeartRate", "✅ 권한 허용 → 심박수 조회")
            refreshHeartRate()
        } else {
            Log.w("HeartRate", "❌ 권한 거부")
            tvHeartRate.text = "권한 거부됨"
            tvHrAlert.text   = "Health Connect 설정 확인 요망"
        }
    }

    // 기본 모니터링(위치, 활동) 권한 요청 런처
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { results ->
        if (results.all { it.value }) startMonitoring()
        else Toast.makeText(this, "필수 권한이 거부되었습니다.", Toast.LENGTH_SHORT).show()
    }

    // BLE 스캔 및 연결 권한 요청 런처 (Android 버전별 분기 처리됨)
    private val blePermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { results ->
        if (results.all { it.value }) {
            // 권한 획득 후 서비스가 바인딩된 상태에서만 스캔 실행
            if (isBound && monitoringService != null) {
                monitoringService!!.startBleScan()
                Toast.makeText(this, "Pico 스캔 중... (10초)", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "서비스 연결 대기 중입니다.", Toast.LENGTH_SHORT).show()
            }
        } else {
            Toast.makeText(this, "BLE 권한이 필요합니다.", Toast.LENGTH_SHORT).show()
        }
    }

    // 백그라운드 서비스 바인딩 콜백
    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName, binder: IBinder) {
            monitoringService = (binder as MonitoringService.LocalBinder).getService()
            isBound = true

            // UI 버튼 상태 동기화
            isMonitoring = monitoringService?.isMonitoringActive ?: false
            updateButtonState()
            updateBleButtonState()

            // 위치 갱신 이벤트 (서버 전송용)
            monitoringService?.onLocationUpdate = { lat, lon, source ->
                sendLocationToServer(lat, lon, source)
            }

            // 걸음 수 갱신 이벤트
            monitoringService?.onStepUpdate = { steps ->
                runOnUiThread { updateStepUI(steps) }
            }

            // 모니터링 종료 이벤트
            monitoringService?.onMonitoringStopped = {
                runOnUiThread {
                    isMonitoring = false
                    updateButtonState()
                    Toast.makeText(this@MainActivity, "모니터링 종료됨", Toast.LENGTH_SHORT).show()
                }
            }

            // IoT 온습도 갱신 이벤트
            monitoringService?.onIoTDataUpdate = { temp, humid, isAlert ->
                runOnUiThread { updateIoTUI(temp, humid, isAlert) }

                // 트래픽 최적화: 온도 0.5°C 또는 습도 2% 이상 변화 시, 혹은 위험 상태일 때만 전송
                val tempChanged  = Math.abs(temp  - lastTemp)  > 0.5f
                val humidChanged = Math.abs(humid - lastHumid) > 2.0f
                if (tempChanged || humidChanged || isAlert) {
                    lastTemp = temp; lastHumid = humid
                    sendIoTDataToServer(temp, humid, isAlert)
                }
            }

            // BLE 연결 상태 변경 이벤트
            monitoringService?.onBleStateUpdate = { connected ->
                runOnUiThread {
                    updateBleButtonState()
                    val msg = if (connected) "Pico 연결됨" else "Pico 연결 해제됨"
                    Toast.makeText(this@MainActivity, msg, Toast.LENGTH_SHORT).show()
                }
            }

            // 충격 감지 이벤트
            monitoringService?.onShockDetected = {
                Log.w("Alert", "충격 감지 → 서버 전송")
                runOnUiThread {
                    Toast.makeText(this@MainActivity, "⚡ 충격 감지! 알림 전송", Toast.LENGTH_SHORT).show()
                }
                sendShockAlertToServer()
            }

            /* STT 데이터 콜백 */
            monitoringService?.onSttDataReady = { wavBytes ->
                Log.d("STT", "WAV 준비: ${wavBytes.size} bytes")
                sendSttToServer(wavBytes)
            }

            // 디버그 패널 갱신 이벤트
            monitoringService?.onDebugUpdate = { info ->
                runOnUiThread {
                    debugDialog?.update(info, lastSentTime)
                }
            }
        }

        override fun onServiceDisconnected(name: ComponentName) {
            isBound = false
        }
    }

    /* ================================================================
     * 생명주기 관리
     * ================================================================ */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initUI()
        initHeartRate()
        initDemoButton()

        // TTS 초기화 코드 추가
        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val result = tts?.setLanguage(Locale.KOREAN)
                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.e("TTS", "기기에서 한국어 음성 데이터를 지원하지 않거나 다운로드가 필요합니다.")
                    runOnUiThread {
                        Toast.makeText(this@MainActivity,
                            "⚠️ 한국어 TTS 데이터 없음. 설정 > 언어 입력에서 다운로드 후 재실행.",
                            Toast.LENGTH_LONG).show()
                    }
                } else {
                    isTtsReady = true // 초기화 완료
                    Log.d("TTS", "TTS 준비 완료")
                }
            } else {
                Log.e("TTS", "TTS 초기화 실패")
            }
        }

        requestBatteryOptimizationExemption()
    }

    override fun onStart() {
        super.onStart()
        // Android 14+ Foreground Service 제한 대응 로직
        // 화면 전환 직후 startForegroundService 호출 시 발생하는 SecurityException 크래시 방지.
        // 여기서는 바인딩(BIND_AUTO_CREATE)만 수행하고, 실제 서비스 시작은 버튼 클릭 이벤트로 지연시킴.
        val intent = Intent(this, MonitoringService::class.java)
        bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
    }

    override fun onStop() {
        super.onStop()
        if (isBound) {
            // 메모리 누수 방지를 위한 콜백 해제
            monitoringService?.apply {
                onLocationUpdate    = null
                onStepUpdate        = null
                onMonitoringStopped = null
                onIoTDataUpdate     = null
                onBleStateUpdate    = null
                onShockDetected     = null
                onSttDataReady      = null
                onDebugUpdate       = null
            }
            unbindService(serviceConnection)
            isBound = false
        }
    }

    override fun onDestroy() {
        tts?.stop()
        tts?.shutdown()
        super.onDestroy()
    }

    /* ================================================================
     * UI 및 리스너 초기화
     * ================================================================ */
    private fun initUI() {
        tvSteps       = findViewById(R.id.tvSteps)
        tvUserName    = findViewById(R.id.tvUserName)
        tvCurrentZone = findViewById(R.id.tvCurrentZone)
        ivProfile     = findViewById(R.id.ivProfile)
        tvTemperature = findViewById(R.id.tvTemperature)
        tvHumidity    = findViewById(R.id.tvHumidity)
        tvHeatAlert   = findViewById(R.id.tvHeatAlert)
        btnStart      = findViewById(R.id.btnStart)
        btnStop       = findViewById(R.id.btnStop)
        btnBleConnect = findViewById(R.id.btnBleConnect)

        // 저장된 유저 정보 세팅
        val prefs    = getSharedPreferences("user_prefs", MODE_PRIVATE)
        val userName = prefs.getString("user_name", "알 수 없음")
        tvUserName.text    = userName
        tvCurrentZone.text = "확인 중..."

        // 프로필 사진 로드
        loadProfileImage(prefs.getString("user_profile_url", "") ?: "")

        // 버튼 리스너 바인딩
        findViewById<Button>(R.id.btnLogout).setOnClickListener { logout() }
        btnStart.setOnClickListener { checkPermissionsAndStart() }

        btnStop.setOnClickListener {
            stopService(Intent(this, MonitoringService::class.java).apply {
                action = "STOP_MONITORING"
            })
            isMonitoring = false
            updateButtonState()
            onZoneExited()   // 구역 상태 리셋
            notifyServerLogout(reason = "모니터링 종료")
            Toast.makeText(this, "모니터링 종료", Toast.LENGTH_SHORT).show()
        }

        btnBleConnect.setOnClickListener {
            if (monitoringService?.isBleConnected == true) {
                monitoringService?.disconnectBle()
                updateBleButtonState()
                Toast.makeText(this, "BLE 연결 해제됨", Toast.LENGTH_SHORT).show()
            } else {
                requestBlePermissions()
            }
        }

        updateButtonState()
    }

    /* ================================================================
     * UI 상태 동기화
     * ================================================================ */
    private fun updateButtonState() {
        btnStart.isEnabled = !isMonitoring
        btnStop.isEnabled  = isMonitoring
        btnStart.alpha     = if (isMonitoring) 0.4f else 1.0f
        btnStop.alpha      = if (isMonitoring) 1.0f else 0.4f
    }

    private fun updateBleButtonState() {
        val connected = monitoringService?.isBleConnected ?: false
        btnBleConnect.text            = if (connected) "Pico 연결 해제" else "Pico 연결"
        btnBleConnect.backgroundTintList = ColorStateList.valueOf(
            if (connected) getColor(android.R.color.holo_red_light)
            else           getColor(android.R.color.holo_blue_light)
        )
    }

    /* ================================================================
     * 권한 확인 및 모니터링 제어
     * ================================================================ */
    private fun checkPermissionsAndStart() {
        val permissions = mutableListOf(
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION,
            Manifest.permission.ACTIVITY_RECOGNITION
        )
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.POST_NOTIFICATIONS)
        }
        requestPermissionLauncher.launch(permissions.toTypedArray())
    }

    private fun requestBlePermissions() {
        val permissions = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            arrayOf(
                Manifest.permission.BLUETOOTH_SCAN,
                Manifest.permission.BLUETOOTH_CONNECT
            )
        } else {
            arrayOf(Manifest.permission.ACCESS_FINE_LOCATION)
        }
        blePermissionLauncher.launch(permissions)
    }

    private fun startMonitoring() {
        if (isMonitoring) return
        isMonitoring = true
        updateButtonState()

        // Android 16+ SecurityException 방어 (버튼 클릭 등 사용자 인터랙션 시점에 실행 보장)
        startForegroundService(Intent(this, MonitoringService::class.java).apply {
            action = "START_MONITORING"
        })

        Toast.makeText(this, "모니터링 시작", Toast.LENGTH_SHORT).show()
    }

    /* ================================================================
     * 서버 데이터 전송 로직 (위치, IoT, 알림)
     * ================================================================ */
    private fun sendLocationToServer(lat: Double, lon: Double, source: String) {
        // PDR 보정 좌표 제외 (순수 GPS 좌표만 전송)
        if (source.startsWith("PDR")) return

        val now      = System.currentTimeMillis()
        val distance = calculateDistance(lastSentLat, lastSentLon, lat, lon)

        // 이동 속도/거리 기반 적응형 전송 간격 세팅 (5초 ~ 30초)
        val interval = when {
            distance > 10.0 -> 5_000L
            distance > 3.0  -> 15_000L
            else            -> 30_000L
        }
        if (now - lastSentTime < interval) return
        if (lastSentTime != 0L && distance < MIN_DISTANCE_METER) return

        val email = getSharedPreferences("user_prefs", MODE_PRIVATE).getString("user_email", "") ?: ""
        if (email.isEmpty()) return

        lastSentTime = now
        lastSentLat  = lat
        lastSentLon  = lon

        lifecycleScope.launch {
            try {
                val response = api.sendLocation(
                    SendLocationRequest(email = email, lat = lat, lon = lon)
                )
                if (response.isSuccessful) {
                    val body = response.body()
                    // 위치 응답에 포함된 구역 정보를 기반으로 알림 상태 갱신
                    runOnUiThread { handleZoneAlert(body?.zone_alert) }
                    Log.d("Location", "위치 전송 성공")
                } else {
                    Log.w("Location", "위치 전송 실패: ${response.code()}")
                }
            } catch (e: Exception) {
                Log.e("Location", "위치 전송 오류: ${e.message}")
            }
        }
    }

    /* ================================================================
     * 위험구역 진입/이탈 상태 관리 및 알림 처리
     * ================================================================ */
    private var currentZoneName: String? = null
    private var zoneAlertJob: kotlinx.coroutines.Job? = null

    private fun handleZoneAlert(alert: ZoneAlert?) {
        // 위험구역(danger)이 아닐 경우 알림을 울리지 않도록 null 처리하여 안전구역으로 간주
        val targetAlert = if (alert?.zone_type == "danger") alert else null

        when {
            // 케이스 1: 위험구역 신규 진입 또는 다른 위험구역으로 변경
            targetAlert != null && targetAlert.zone_name != currentZoneName -> {
                Log.d("Zone", "위험구역 진입/변경: $currentZoneName → ${targetAlert.zone_name}")
                currentZoneName = targetAlert.zone_name
                onZoneEntered(targetAlert)
            }
            // 케이스 2: 같은 위험구역 유지 — 코루틴 이미 작동 중
            targetAlert != null && targetAlert.zone_name == currentZoneName -> { }
            // 케이스 3: 위험구역 완전 이탈
            targetAlert == null && currentZoneName != null -> {
                Log.d("Zone", "위험구역 이탈: $currentZoneName")
                onZoneExited()
            }
        }
    }

    private fun onZoneEntered(alert: ZoneAlert) {
        tvCurrentZone.text = alert.zone_name

        // 위험구역 최초 진입 경고 다이얼로그 고정 노출
        android.app.AlertDialog.Builder(this)
            .setTitle("🚨 ${alert.zone_name}")
            .setMessage(alert.message)
            .setPositiveButton("확인", null)
            .setCancelable(false)
            .show()

        showZoneNotification(alert)

        // 5초 주기 위험구역 경고 TTS 및 진동
        val ttsMessage = "${alert.zone_name} 위험구역입니다. 즉시 이동하시기 바랍니다."

        zoneAlertJob?.cancel()
        zoneAlertJob = lifecycleScope.launch {
            while (true) {
                vibrateForZone() // 파라미터 제거됨
                if (isTtsReady) {
                    tts?.speak(ttsMessage, TextToSpeech.QUEUE_FLUSH, null, "ZoneAlertTTS")
                }
                kotlinx.coroutines.delay(5000L)
            }
        }
    }

    private fun onZoneExited() {
        zoneAlertJob?.cancel()
        zoneAlertJob = null
        currentZoneName = null

        // TTS 즉시 중단
        tts?.stop()

        tvCurrentZone.text = "안전 구역"

        // 상태바 알림 즉시 제거
        getSystemService(NotificationManager::class.java).cancel(ZONE_NOTIFICATION_ID)

        Toast.makeText(this, "✅ 안전 구역으로 복귀했습니다", Toast.LENGTH_SHORT).show()
    }

    private fun vibrateForZone() {
        val vibrator = getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            // 위험구역 전용 강한 진동 패턴
            val pattern = VibrationEffect.createWaveform(
                longArrayOf(0, 500, 200, 500, 200, 500), -1
            )
            vibrator.vibrate(pattern)
        }
    }

    // 구역 알림 갱신 및 중복 생성 방지용 고정 ID
    private val ZONE_NOTIFICATION_ID = 9001

    private fun showZoneNotification(alert: ZoneAlert) {
        val channelId = "zone_alert_channel"

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                channelId, "구역 경고",
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                enableVibration(true)
                setShowBadge(true)
            }
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }

        val notification = NotificationCompat.Builder(this, channelId)
            .setContentTitle("⚠️ ${alert.zone_name} 진입")
            .setContentText(alert.message)
            .setSmallIcon(android.R.drawable.ic_dialog_alert) // 아이콘 고정
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setOngoing(true)
            .build()

        getSystemService(NotificationManager::class.java).notify(ZONE_NOTIFICATION_ID, notification)
    }

    /* ================================================================
     * 기타 서버 연동 (IoT, 충격, STT)
     * ================================================================ */
    private fun sendIoTDataToServer(temp: Float, humid: Float, isAlert: Boolean) {
        val email = getSharedPreferences("user_prefs", MODE_PRIVATE).getString("user_email", "") ?: ""
        if (email.isEmpty()) return

        lifecycleScope.launch {
            try {
                val response = api.sendIoTData(
                    IoTDataRequest(email = email, temperature = temp, humidity = humid, heat_alert = isAlert)
                )
                if (response.isSuccessful) Log.d("IoT", "온습도 전송 성공")
                else Log.w("IoT", "온습도 전송 실패: ${response.code()}")
            } catch (e: Exception) { Log.e("IoT", "온습도 전송 오류: ${e.message}") }
        }
    }

    private fun sendShockAlertToServer() {
        val email = getSharedPreferences("user_prefs", MODE_PRIVATE).getString("user_email", "") ?: ""
        if (email.isEmpty()) return
        lifecycleScope.launch {
            try {
                api.sendEmergencySos(SosRequest(email, "IMPACT", "안전모 강한 충격 감지"))
                Log.d("Shock", "충격 알림 전송 성공")
            } catch (e: Exception) { Log.e("Shock", "전송 오류: ${e.message}") }
        }
    }

    private fun sendSttToServer(wavBytes: ByteArray) {
        val prefs = getSharedPreferences("user_prefs", MODE_PRIVATE)
        val email = prefs.getString("user_email", "") ?: ""
        if (email.isEmpty()) return

        val lat = prefs.getFloat("last_lat", 0f).toDouble()
        val lon = prefs.getFloat("last_lon", 0f).toDouble()

        lifecycleScope.launch {
            try {
                // 임시 WAV 파일 생성 후 Multipart 업로드
                val wavFile = java.io.File(cacheDir, "stt_${System.currentTimeMillis()}.wav")
                wavFile.writeBytes(wavBytes)

                val response = api.sendSttAlert(
                    email.toRequestBody("text/plain".toMediaType()),
                    lat.toString().toRequestBody("text/plain".toMediaType()),
                    lon.toString().toRequestBody("text/plain".toMediaType()),
                    okhttp3.MultipartBody.Part.createFormData(
                        "audio", wavFile.name,
                        wavFile.asRequestBody("audio/wav".toMediaType())
                    )
                )
                wavFile.delete() // 전송 후 찌꺼기 파일 삭제

                if (response.isSuccessful) {
                    val body = response.body()
                    Log.d("STT", "결과: is_sos=${body?.is_sos}, 인식=${body?.message}")
                    if (body?.is_sos == true) {
                        api.sendEmergencySos(SosRequest(email, "VOICE_SOS", "작업자 구조 요청: ${body.message}"))
                        runOnUiThread {
                            Toast.makeText(this@MainActivity, "🆘 구조 요청 전송됨: \"${body.message}\"", Toast.LENGTH_LONG).show()
                        }
                    }
                } else Log.w("STT", "STT 전송 실패: ${response.code()}")
            } catch (e: Exception) { Log.e("STT", "전송 오류: ${e.message}") }
        }
    }

    // Haversine 공식 기반 두 좌표 간 거리(미터) 계산
    private fun calculateDistance(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
        if (lat1 == 0.0 && lon1 == 0.0) return Double.MAX_VALUE
        val r    = 6371000.0
        val dLat = Math.toRadians(lat2 - lat1)
        val dLon = Math.toRadians(lon2 - lon1)
        val a    = Math.sin(dLat / 2).pow(2) +
                Math.cos(Math.toRadians(lat1)) * Math.cos(Math.toRadians(lat2)) * Math.sin(dLon / 2).pow(2)
        return r * 2 * Math.asin(Math.sqrt(a))
    }

    /* ================================================================
     * 로그아웃 처리
     * ================================================================ */
    private fun logout() {
        val email = getSharedPreferences("user_prefs", MODE_PRIVATE).getString("user_email", "") ?: ""

        // 구역 알림 및 서비스 초기화
        onZoneExited()
        if (isMonitoring) {
            stopService(Intent(this, MonitoringService::class.java).apply { action = "STOP_MONITORING" })
        }
        monitoringService?.disconnectBle()

        // 서버 세션 해제 및 클라이언트 로컬 데이터 소거 (UI 멈춤 방지를 위한 3초 타임아웃 적용)
        lifecycleScope.launch {
            if (email.isNotEmpty()) {
                try {
                    val response = withTimeoutOrNull(3000L) { api.logout(mapOf("email" to email)) }
                    if (response?.isSuccessful == true) Log.d("Logout", "✅ 서버 로그아웃 성공")
                    else Log.w("Logout", "⚠️ 서버 로그아웃 실패/시간초과 (5분 후 서버 자체 정리)")
                } catch (e: Exception) { Log.e("Logout", "서버 로그아웃 오류: ${e.message}") }
            }

            getSharedPreferences("user_prefs", MODE_PRIVATE).edit().clear().apply()
            startActivity(Intent(this@MainActivity, LoginActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
            })
            finish()
        }
    }

    // 모니터링 강제 종료 시 서버 측 지도 마커 제거를 위한 통보 (Fire-and-Forget)
    private fun notifyServerLogout(reason: String) {
        val email = getSharedPreferences("user_prefs", MODE_PRIVATE).getString("user_email", "") ?: ""
        if (email.isEmpty()) return

        lifecycleScope.launch {
            try {
                val response = withTimeoutOrNull(3000L) { api.logout(mapOf("email" to email)) }
                if (response?.isSuccessful == true) Log.d("Logout", "✅ [$reason] 서버 통보 성공")
                else Log.w("Logout", "⚠️ [$reason] 서버 통보 실패")
            } catch (e: Exception) { Log.e("Logout", "[$reason] 서버 통보 오류: ${e.message}") }
        }
    }

    /* ================================================================
     * 기타 UI 업데이트 및 모듈 초기화
     * ================================================================ */
    private fun updateStepUI(steps: Int) {
        tvSteps.text = "$steps"
    }

    // Glide를 활용한 서버 프로필 이미지 렌더링 (경로 중복 '/' 방지 처리 포함)
    private fun loadProfileImage(profileUrl: String) {
        if (profileUrl.isNotEmpty()) {
            val base     = ApiService.BASE_URL.trimEnd('/')
            val path     = if (profileUrl.startsWith("/")) profileUrl else "/$profileUrl"
            val fullUrl  = "$base$path"

            com.bumptech.glide.Glide.with(this)
                .load(fullUrl)
                .centerCrop()
                .placeholder(R.mipmap.ic_launcher)
                .error(R.mipmap.ic_launcher)
                .into(ivProfile)
        } else {
            ivProfile.setImageResource(R.mipmap.ic_launcher)
        }
    }

    private fun updateIoTUI(temp: Float, humid: Float, isAlert: Boolean) {
        tvTemperature.text = String.format("온도: %.1f°C", temp)
        tvHumidity.text    = String.format("습도: %.1f%%", humid)

        if (isAlert) {
            tvHeatAlert.text = "⚠️ 열사병 위험! 즉시 휴식"
            tvHeatAlert.setTextColor(getColor(android.R.color.holo_red_dark))

            // 진동 알림 가동
            val vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                (getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as android.os.VibratorManager).defaultVibrator
            } else {
                @Suppress("DEPRECATION")
                getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                vibrator.vibrate(VibrationEffect.createWaveform(longArrayOf(0, 500, 200, 500), -1))
            }
        } else {
            tvHeatAlert.text = "정상 범위"
            tvHeatAlert.setTextColor(getColor(android.R.color.holo_green_dark))
        }
        updateBleButtonState()
    }

    // 심박수 매니저(Health Connect) 초기화 및 UI 바인딩
    private fun initHeartRate() {
        tvHeartRate  = findViewById(R.id.tvHeartRate)
        tvHrAlert    = findViewById(R.id.tvHrAlert)
        btnHrRefresh = findViewById(R.id.btnHrRefresh)

        btnHrRefresh.setOnClickListener { checkAndReadHeartRate() }

        if (heartRateManager.isAvailable()) {
            checkAndReadHeartRate()
            startHeartRatePolling() // 1분 주기 자동 수집
        } else {
            tvHeartRate.text = "Health Connect 미설치"
            tvHrAlert.text   = "Play 스토어 설치 요망"
        }
    }

    private fun checkAndReadHeartRate() {
        lifecycleScope.launch {
            if (!heartRateManager.hasPermissions()) hrPermissionLauncher.launch(heartRateManager.permissions)
            else refreshHeartRate()
        }
    }

    private fun refreshHeartRate() {
        lifecycleScope.launch {
            when (val result = heartRateManager.readRecentHeartRate(minutesAgo = 30)) {
                is HeartRateResult.Success -> updateHeartRateUI(result)
                is HeartRateResult.NoData  -> {
                    tvHeartRate.text = "심박수: 데이터 없음"
                    tvHrAlert.text   = "측정 필요"
                }
                is HeartRateResult.Error   -> {
                    tvHeartRate.text = "조회 실패"
                    tvHrAlert.text   = result.msg
                }
            }
        }
    }

    private fun updateHeartRateUI(result: HeartRateResult.Success) {
        tvHeartRate.text = "심박수: ${result.latestBpm} bpm"
        tvHrAlert.text   = result.alert.label
        tvHrAlert.setTextColor(result.alert.color)

        if (result.alert == HeartRateAlert.TOO_HIGH || result.alert == HeartRateAlert.TOO_LOW) {
            triggerHeartRateAlert(result)
        }
        sendHeartRateToServer(result.latestBpm, result.avgBpm, result.alert.name)
    }

    private fun triggerHeartRateAlert(result: HeartRateResult.Success) {
        val vibrator = getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator.vibrate(VibrationEffect.createWaveform(longArrayOf(0, 500, 200, 500, 200, 500), -1))
        }
        Toast.makeText(this, "⚠️ 심박수 이상 감지: ${result.latestBpm}bpm", Toast.LENGTH_LONG).show()
    }

    private fun startHeartRatePolling() {
        lifecycleScope.launch {
            while (true) {
                kotlinx.coroutines.delay(60_000L)
                if (isMonitoring) refreshHeartRate()
            }
        }
    }

    private fun sendHeartRateToServer(latest: Long, avg: Long, status: String) {
        val email = getSharedPreferences("user_prefs", MODE_PRIVATE).getString("user_email", "") ?: ""
        if (email.isEmpty()) return

        lifecycleScope.launch {
            try {
                api.sendHeartRate(HeartRateRequest(email = email, bpm = latest, avg_bpm = avg, status = status))
            } catch (e: Exception) { Log.e("HeartRate", "서버 전송 오류") }
        }
    }

    /* ================================================================
     * 시연용 데모 컨트롤러 바인딩
     * ================================================================ */
    private fun initDemoButton() {
        val btnDemo = findViewById<android.widget.Button>(R.id.btnDemo) ?: return
        if (DemoController.isEnabled(this)) {
            btnDemo.visibility = android.view.View.VISIBLE
            btnDemo.setOnClickListener { showDemoPanel() }
            btnDemo.setOnLongClickListener {
                debugDialog = DebugPanelDialog(this)
                debugDialog?.show()
                true
            }
        } else {
            btnDemo.visibility = android.view.View.GONE
        }
    }

    private fun showDemoPanel() {
        DemoPanelDialog(this, object : DemoPanelDialog.Callbacks {
            override fun onHeartRateInject(bpm: Long) {
                updateHeartRateUI(makeHrResult(bpm))
                Toast.makeText(this@MainActivity, "🎬 심박수 주입", Toast.LENGTH_SHORT).show()
            }
            override fun onLocationInject(lat: Double, lon: Double, label: String) {
                lastSentTime = 0L
                sendLocationToServer(lat, lon, "GPS")
                Toast.makeText(this@MainActivity, "🎬 위치 주입: $label", Toast.LENGTH_SHORT).show()
            }
            override fun onIoTInject(temp: Float, humid: Float, alert: Boolean) {
                updateIoTUI(temp, humid, alert)
                sendIoTDataToServer(temp, humid, alert)
            }
        }).show()
    }

    private fun makeHrResult(bpm: Long): HeartRateResult.Success {
        val alert = when {
            bpm < 50  -> HeartRateAlert.TOO_LOW
            bpm > 120 -> HeartRateAlert.TOO_HIGH
            bpm > 100 -> HeartRateAlert.HIGH
            else      -> HeartRateAlert.NORMAL
        }
        return HeartRateResult.Success(
            latestBpm  = bpm,
            avgBpm     = bpm,
            alert      = alert,
            latestTime = java.time.Instant.now()
        )
    }

    // 백그라운드 모니터링 유지를 위한 배터리 최적화 제외 권한 요청 (Doze 모드 방어)
    private fun requestBatteryOptimizationExemption() {
        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        if (!pm.isIgnoringBatteryOptimizations(packageName)) {
            val intent = Intent().apply {
                action = Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS
                data = Uri.parse("package:$packageName")
            }
            startActivity(intent)
        }
    }
}