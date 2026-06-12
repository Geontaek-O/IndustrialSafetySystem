package com.example.ConSAFE

import android.annotation.SuppressLint
import android.app.*
import android.bluetooth.*
import android.bluetooth.le.*
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.hardware.*
import android.os.*
import android.util.Log
import androidx.core.app.NotificationCompat
import com.google.android.gms.location.*
import java.util.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/**
 * [MonitoringService]
 * 실시간 GPS, 센서(PDR), BLE 연동 및 상태 모니터링 백그라운드 포그라운드 서비스.
 */
class MonitoringService : Service(), SensorEventListener {

    /* --- 로컬 바인더 --- */
    private val binder = LocalBinder()
    inner class LocalBinder : Binder() {
        fun getService(): MonitoringService = this@MonitoringService
    }
    override fun onBind(intent: Intent): IBinder = binder

    /* --- GPS 연동 --- */
    private lateinit var fusedLocationClient: FusedLocationProviderClient
    private lateinit var locationCallback: LocationCallback
    private val kalmanFilter    = GpsKalmanFilter()
    private var isGpsInitialized = false
    private var lastGpsTimestamp = 0L

    /* --- 디바이스 센서 (걸음수, 가속도, 지자기) --- */
    private lateinit var sensorManager: SensorManager
    private var stepCounterSensor:   Sensor? = null
    private var accelerometerSensor: Sensor? = null
    private var magneticSensor:      Sensor? = null

    val accelerometerReading = FloatArray(3)
    val magnetometerReading  = FloatArray(3)

    /* --- PDR (보행자 추측 항법) 상태 관리 --- */
    var currentLat       = 0.0
    var currentLon       = 0.0
    var currentAzimuth   = 0f
    var currentSteps     = 0
    var initialStepCount = -1f
    private var smoothedAzimuth = 0f
    private val ALPHA = 0.05f

    /* --- 서비스 구동 상태 --- */
    var isMonitoringActive  = false
    private var isForegroundStarted = false

    /* --- UI 연동 콜백 이벤트 --- */
    var onLocationUpdate:    ((Double, Double, String) -> Unit)? = null
    var onStepUpdate:        ((Int) -> Unit)?                    = null
    var onHeadingUpdate:     ((Float) -> Unit)?                  = null
    var onMonitoringStopped: (() -> Unit)?                       = null
    var onIoTDataUpdate:     ((Float, Float, Boolean) -> Unit)?  = null
    var onBleStateUpdate:    ((Boolean) -> Unit)?                = null
    var onShockDetected:     (() -> Unit)?                       = null  // 충격 감지 발생 시
    var onSttDataReady:      ((ByteArray) -> Unit)?              = null  // STT WAV 바이너리 준비 완료 시
    var onDebugUpdate:       ((DebugInfo) -> Unit)?              = null  // 디버그 오버레이 갱신용

    /* --- BLE 통신 --- */
    private var bluetoothGatt: BluetoothGatt? = null
    private var bleScanner:    BluetoothLeScanner? = null
    var isBleConnected = false

    // Pico 펌웨어 연동용 UUID 선언 (펌웨어와 1:1 매칭 필수)
    private val PICO_SERVICE_UUID = UUID.fromString("12345678-1234-1234-1234-123456789abc")
    private val DHT22_CHAR_UUID   = UUID.fromString("12345678-1234-1234-1234-123456789001")
    private val MIC_CHAR_UUID     = UUID.fromString("12345678-1234-1234-1234-123456789002")
    private val VIB_CHAR_UUID     = UUID.fromString("12345678-1234-1234-1234-123456789003")
    private val NOTIFY_DESCRIPTOR = UUID.fromString("00002902-0000-1000-8000-00805f9b34fb")

    // 마이크 오디오 버퍼 설정 (16kHz, 16bit, Mono 기준 3초 분량)
    private val micBuffer      = java.io.ByteArrayOutputStream()
    private val MIC_BUFFER_MAX = 16000 * 2 * 3
    private val micBufferLock  = Any()

    // 열사병 위험 임계값 설정
    private val TEMP_ALERT_THRESHOLD  = 35.0f
    private val HUMID_ALERT_THRESHOLD = 60.0f

    // 서비스 전용 비동기 코루틴 스코프 (BLE 백그라운드 재스캔용)
    private val serviceScope = CoroutineScope(Dispatchers.Main + SupervisorJob())

    /* --- 알림 설정 --- */
    private val CHANNEL_ID    = "monitoring_channel"
    private val NOTIFICATION_ID = 1

    /* ================================================================
     * 서비스 생명주기
     * ================================================================ */
    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()

        // startForeground() 선제 호출 배제.
        // - BIND_AUTO_CREATE 생성 시점의 Activity 비정상 상태(Resume 이전)에 따른
        //   Android 14+ FGS SecurityException 크래시 사전 차단.
        // - 실제 사용자 진입(startMonitoringInternal) 시점에만 명시적 호출.
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            "START_MONITORING" -> startMonitoringInternal()
            "STOP_MONITORING"  -> stopMonitoringInternal()
            null -> {
                // 시스템 메모리 부족에 의한 서비스 강제 재시작(START_STICKY) 대응.
                // 이전 포그라운드 상태 복구를 위해 알림 즉시 재등록 (ANR 방어).
                startForeground(NOTIFICATION_ID, buildNotification(isMonitoring = isMonitoringActive))
                Log.d("MonitoringService", "START_STICKY 재시작 — 알림 재등록")
            }
        }
        return START_STICKY
    }

    /** * 모니터링 강제 시작 (MainActivity Binder 경유 직접 호출).
     */
    fun startMonitoringInternal() {
        if (isMonitoringActive) return
        isMonitoringActive = true

        // 사용자 명시적 조작에 의한 Activity 완전 포그라운드 상태.
        // 이 시점에 FGS 알림을 발생시켜 SecurityException 조건 우회 충족.
        startForegroundNotification()
        initSensors()
        initGPS()
        updateNotification()

        tryDirectConnect()
    }

    /** * 모니터링 강제 종료 (MainActivity Binder 경유 직접 호출).
     */
    fun stopMonitoringInternal() {
        isMonitoringActive = false
        isForegroundStarted = false
        stopMonitoring()
        onMonitoringStopped?.invoke()
        stopSelf()
    }

    override fun onDestroy() {
        super.onDestroy()
        serviceScope.cancel()   // 코루틴 메모리 누수 방지
        stopMonitoring()
        disconnectBle()
    }

    /* ================================================================
     * 포그라운드 상태 및 알림 제어
     * ================================================================ */

    /**
     * 포그라운드 서비스(FGS) 승격 및 알림 고정.
     * Android 버전에 따른 ForegroundServiceType 권한 분기 포함.
     */
    private fun startForegroundNotification() {
        if (isForegroundStarted) return
        isForegroundStarted = true
        val notification = buildNotification(isMonitoring = true)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(
                NOTIFICATION_ID, notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_LOCATION or ServiceInfo.FOREGROUND_SERVICE_TYPE_CONNECTED_DEVICE
            )
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NOTIFICATION_ID, notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_LOCATION
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    fun promoteToLocation() { /* startForegroundNotification 단일 처리로 통합 */ }
    fun promoteForegroundToBle() { /* Location 타입 FGS 선언으로 BLE 스캔 동시 허용됨 */ }

    private fun updateNotification() {
        val notification = buildNotification(isMonitoring = isMonitoringActive)
        getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID, notification)
    }

    private fun buildNotification(isMonitoring: Boolean): Notification {
        val openAppIntent = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
            },
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val stopIntent = PendingIntent.getService(
            this, 0,
            Intent(this, MonitoringService::class.java).apply { action = "STOP_MONITORING" },
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(if (isMonitoring) "모니터링 실행 중..." else "안전 모니터링")
            .setContentText(
                if (isMonitoring) "GPS 및 걸음 수 추적 중입니다. 탭하여 앱으로 돌아가기"
                else "모니터링이 준비되었습니다."
            )
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setContentIntent(openAppIntent)
            .setOngoing(true)
            .apply {
                if (isMonitoring) addAction(android.R.drawable.ic_delete, "종료", stopIntent)
            }
            .build()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID, "위치 모니터링",
                NotificationManager.IMPORTANCE_LOW
            )
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }
    }

    /* ================================================================
     * GPS 위치 트래킹 로직
     * ================================================================ */
    @SuppressLint("MissingPermission")
    private fun initGPS() {
        if (isGpsInitialized) return
        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)
        locationCallback = object : LocationCallback() {
            override fun onLocationResult(result: LocationResult) {
                result.locations.lastOrNull()?.let { loc ->
                    val now     = System.currentTimeMillis()
                    val elapsed = if (lastGpsTimestamp == 0L) 1000L else now - lastGpsTimestamp
                    lastGpsTimestamp = now

                    // 최초 측정 시 오차 허용치(60m) 완화, 이후 정밀도 상향(25m) 요구
                    val threshold = if (currentLat == 0.0) 60f else 25f
                    if (loc.accuracy <= threshold) {

                        // 칼만 필터를 통한 노이즈 스무딩 및 보정 좌표 산출
                        val (filteredLat, filteredLon) = kalmanFilter.process(
                            loc.latitude, loc.longitude, loc.accuracy,
                            isMoving  = isDeviceMoving(),
                            elapsedMs = elapsed
                        )
                        currentLat = filteredLat
                        currentLon = filteredLon

                        onLocationUpdate?.invoke(currentLat, currentLon, "GPS (±${loc.accuracy.toInt()}m)")
                    }
                }
            }
        }

        // GPS 요청 최적화 (배터리/정확도 타협 설정)
        val request = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 3000L)
            .setMinUpdateDistanceMeters(0.5f)
            .setMinUpdateIntervalMillis(1000L)
            .setMaxUpdateDelayMillis(5000L)
            .setWaitForAccurateLocation(true)
            .build()

        fusedLocationClient.requestLocationUpdates(request, locationCallback, Looper.getMainLooper())
        isGpsInitialized = true
    }

    /* ================================================================
     * 하드웨어 센서 수집 및 PDR(보행자 추측 항법) 연산
     * ================================================================ */
    private fun initSensors() {
        sensorManager       = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        stepCounterSensor   = sensorManager.getDefaultSensor(Sensor.TYPE_STEP_COUNTER)
        accelerometerSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        magneticSensor      = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)

        stepCounterSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST) }
        accelerometerSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_UI) }
        magneticSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_UI) }
    }

    override fun onSensorChanged(event: SensorEvent?) {
        val e = event ?: return
        when (e.sensor.type) {
            Sensor.TYPE_STEP_COUNTER -> {
                // 걸음 수 업데이트 확인 시점마다 PDR 예측 연산 트리거
                if (initialStepCount < 0) initialStepCount = e.values[0]
                val newSteps = (e.values[0] - initialStepCount).toInt()
                if (newSteps > currentSteps) {
                    currentSteps = newSteps
                    onStepUpdate?.invoke(currentSteps)
                    calculatePDR()
                }
            }
            Sensor.TYPE_ACCELEROMETER -> System.arraycopy(e.values, 0, accelerometerReading, 0, 3)
            Sensor.TYPE_MAGNETIC_FIELD -> {
                System.arraycopy(e.values, 0, magnetometerReading, 0, 3)
                updateAzimuth()
            }
        }
    }

    /**
     * 가속도 및 지자기 센서 융합 기반 스마트폰 방위각(Heading) 산출
     */
    private fun updateAzimuth() {
        val rotMatrix      = FloatArray(9)
        val remappedMatrix = FloatArray(9)
        val orientation    = FloatArray(3)

        if (!SensorManager.getRotationMatrix(rotMatrix, null, accelerometerReading, magnetometerReading)) return

        // 폰 파지 상태(피치각) 기준 좌표축 재설정 (수직/수평 방어)
        val pitch = Math.toDegrees(Math.atan2(accelerometerReading[1].toDouble(), accelerometerReading[2].toDouble()))
        val (axisX, axisY) = if (Math.abs(pitch) > 45) {
            Pair(SensorManager.AXIS_X, SensorManager.AXIS_Z)
        } else {
            Pair(SensorManager.AXIS_X, SensorManager.AXIS_MINUS_Z)
        }

        SensorManager.remapCoordinateSystem(rotMatrix, axisX, axisY, remappedMatrix)
        SensorManager.getOrientation(remappedMatrix, orientation)

        // 방위각 360도 보정 및 스무딩 필터 적용
        val raw = ((Math.toDegrees(orientation[0].toDouble()) + 360) % 360).toFloat()
        val delta = raw - smoothedAzimuth
        val wrapped = when {
            delta > 180f  -> delta - 360f
            delta < -180f -> delta + 360f
            else          -> delta
        }
        smoothedAzimuth = (smoothedAzimuth + ALPHA * wrapped + 360f) % 360f
        currentAzimuth  = smoothedAzimuth
        onHeadingUpdate?.invoke(currentAzimuth)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    /** 가속도 벡터 크기 기반 단말 이동 상태 판별 (정지/이동) */
    private fun isDeviceMoving(): Boolean {
        val mag = Math.sqrt(accelerometerReading.sumOf { (it * it).toDouble() })
        return Math.abs(mag - 9.81) > 1.5
    }

    /** 걸음 발생 시 PDR 공식을 통한 위경도 좌표 예측 오프셋 계산 */
    private fun calculatePDR() {
        if (currentLat == 0.0 && currentLon == 0.0) return
        if (!isDeviceMoving()) return

        val stepLen = calculateStepLength()
        val rad     = Math.toRadians(currentAzimuth.toDouble())

        // 위도/경도 1도당 미터 거리 환산 적용 (약 111km)
        currentLat += (stepLen * Math.cos(rad)) / 111111.0
        currentLon += (stepLen * Math.sin(rad)) / (111111.0 * Math.cos(Math.toRadians(currentLat)))

        onLocationUpdate?.invoke(currentLat, currentLon, "PDR")
    }

    /** 가속도 진폭 기반 동적 보폭(Stride Length) 추정 */
    private fun calculateStepLength(): Double {
        val mag = Math.sqrt(accelerometerReading.sumOf { (it * it).toDouble() })
        return when {
            mag < 9.0  -> 0.55
            mag < 10.5 -> 0.70
            mag < 12.0 -> 0.85
            else       -> 1.00
        }
    }

    private fun stopMonitoring() {
        if (::sensorManager.isInitialized) sensorManager.unregisterListener(this)
        if (isGpsInitialized) {
            fusedLocationClient.removeLocationUpdates(locationCallback)
            isGpsInitialized = false
        }
    }

    /* ================================================================
     * BLE 통신 로직
     * ================================================================ */
    @SuppressLint("MissingPermission")
    fun startBleScan() {
        promoteForegroundToBle()

        val manager = getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
        val adapter = manager.adapter

        // 블루투스 비활성화 상태 예외 처리
        if (adapter == null || !adapter.isEnabled) {
            Log.e("BLE", "블루투스가 꺼져 있습니다.")
            return
        }

        bleScanner = adapter.bluetoothLeScanner
        if (bleScanner == null) {
            Log.e("BLE", "BLE 스캐너 획득 실패.")
            return
        }

        Log.d("BLE", "전체 BLE 기기 스캔 시작")

        val settings = ScanSettings.Builder().setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY).build()
        bleScanner?.startScan(null, settings, scanCallback)

        // 15초 스캔 타임아웃 지정
        Handler(Looper.getMainLooper()).postDelayed({
            stopBleScan()
            Log.d("BLE", "15초 경과 — 스캔 종료")
        }, 15000L)
    }

    @SuppressLint("MissingPermission")
    fun stopBleScan() {
        bleScanner?.stopScan(scanCallback)
    }

    private val scanCallback = object : ScanCallback() {
        @SuppressLint("MissingPermission")
        override fun onScanResult(callbackType: Int, result: ScanResult) {
            val deviceName = result.device.name ?: ""
            Log.d("BLE", "기기 감지: $deviceName / ${result.device.address}")

            // 타겟 디바이스 네임 필터링
            if (deviceName != "IoTNode_A3F") return

            Log.d("BLE", "Pico 발견. 연결 진입.")
            stopBleScan()

            // 로컬 MAC 저장
            val prefs = getSharedPreferences("ble_prefs", Context.MODE_PRIVATE)
            prefs.edit().putString("pico_mac", result.device.address).apply()

            // OS 레벨 자동 재연결 플래그(autoConnect = true) 부여
            Handler(Looper.getMainLooper()).post {
                result.device.connectGatt(
                    this@MonitoringService,
                    true,
                    gattCallback,
                    BluetoothDevice.TRANSPORT_LE
                )
            }
        }

        override fun onScanFailed(errorCode: Int) {
            val reason = when (errorCode) {
                1 -> "이미 스캔 중"
                2 -> "앱 등록 한도 초과"
                3 -> "내부 오류"
                4 -> "BLE 기능 미지원"
                else -> "알 수 없음"
            }
            Log.e("BLE", "스캔 실패: $errorCode ($reason)")
        }
    }

    @SuppressLint("MissingPermission")
    private val gattCallback = object : BluetoothGattCallback() {

        override fun onConnectionStateChange(gatt: BluetoothGatt, status: Int, newState: Int) {
            when (newState) {
                BluetoothProfile.STATE_CONNECTED -> {
                    isBleConnected = true
                    bluetoothGatt  = gatt
                    Log.d("BLE", "Pico 연결 성공. 서비스 탐색 대기.")
                    onBleStateUpdate?.invoke(true)

                    currentDebug = currentDebug.copy(bleConnected = true, lastEvent = "BLE 연결됨")
                    onDebugUpdate?.invoke(currentDebug)

                    // 연결 직후 GATT 캐시 충돌 방지를 위한 Refresh 호출
                    Handler(Looper.getMainLooper()).postDelayed({
                        try {
                            val refresh = BluetoothGatt::class.java.getMethod("refresh")
                            val result = refresh.invoke(gatt) as Boolean
                            Log.d("BLE", "GATT 캐시 리프레시: $result")
                        } catch (e: Exception) {
                            Log.w("BLE", "GATT 리프레시 미지원 기기: ${e.message}")
                        }

                        // 캐시 정리 후 서비스 탐색 수행
                        Handler(Looper.getMainLooper()).postDelayed({
                            val success = gatt.discoverServices()
                            Log.d("BLE", "discoverServices 결과: $success")
                        }, 500L)
                    }, 1000L)
                }
                BluetoothProfile.STATE_DISCONNECTED -> {
                    isBleConnected = false
                    bluetoothGatt?.close()
                    bluetoothGatt  = null
                    Log.d("BLE", "Pico 연결 유실")
                    onBleStateUpdate?.invoke(false)

                    // 배터리 최적화 환경에서 Handler 누락 방지를 위한 코루틴 재스캔 루틴
                    serviceScope.launch {
                        delay(5000L)
                        if (!isBleConnected) {
                            Log.d("BLE", "백그라운드 자동 재스캔 시도")
                            startBleScan()
                        }
                    }
                }
            }
        }

        override fun onServicesDiscovered(gatt: BluetoothGatt, status: Int) {
            if (status != BluetoothGatt.GATT_SUCCESS) {
                Log.e("BLE", "서비스 탐색 실패($status). 캐시 리프레시 및 재시도.")
                try {
                    val refresh = BluetoothGatt::class.java.getMethod("refresh")
                    refresh.invoke(gatt)
                } catch (e: Exception) { }
                Handler(Looper.getMainLooper()).postDelayed({ gatt.discoverServices() }, 1000L)
                return
            }

            val service = gatt.getService(PICO_SERVICE_UUID) ?: run {
                Log.w("BLE", "Pico 서비스 불일치. UUID 확인 요망.")
                return
            }

            val dhtChar = service.getCharacteristic(DHT22_CHAR_UUID)
            val micChar = service.getCharacteristic(MIC_CHAR_UUID)
            val vibChar = service.getCharacteristic(VIB_CHAR_UUID)

            // 센서별 Notification 데스크립터 개별 활성화 (딜레이 순차 적용)
            if (dhtChar != null) {
                gatt.setCharacteristicNotification(dhtChar, true)
                Handler(Looper.getMainLooper()).postDelayed({ enableNotify(gatt, dhtChar, "DHT22") }, 100L)
            } else { Log.w("BLE", "DHT22 특성 미발견") }

            if (micChar != null) {
                gatt.setCharacteristicNotification(micChar, true)
                Handler(Looper.getMainLooper()).postDelayed({ enableNotify(gatt, micChar, "MIC") }, 300L)
            } else { Log.w("BLE", "MIC 특성 미발견") }

            if (vibChar != null) {
                gatt.setCharacteristicNotification(vibChar, true)
                Handler(Looper.getMainLooper()).postDelayed({ enableNotify(gatt, vibChar, "VIB") }, 500L)
            } else { Log.w("BLE", "VIB 특성 미발견") }
        }

        // Android 12 이하 호환용 콜백
        @Suppress("DEPRECATION")
        override fun onCharacteristicChanged(
            gatt: BluetoothGatt,
            characteristic: BluetoothGattCharacteristic
        ) {
            @Suppress("DEPRECATION")
            val bytes = characteristic.value ?: return
            handleCharacteristic(characteristic.uuid, bytes)
        }

        // Android 13 이상 콜백
        override fun onCharacteristicChanged(
            gatt: BluetoothGatt,
            characteristic: BluetoothGattCharacteristic,
            value: ByteArray
        ) {
            handleCharacteristic(characteristic.uuid, value)
        }
    }

    /** 수신된 BLE 특성(Characteristic) 파싱 분기 처리 */
    private fun handleCharacteristic(uuid: java.util.UUID, value: ByteArray) {
        when (uuid) {
            DHT22_CHAR_UUID -> {
                val raw = value.toString(Charsets.UTF_8)
                parseDht22AndNotify(raw)
            }
            MIC_CHAR_UUID -> {
                // 수신 오디오 청크 RMS 분석을 통한 무음/유의미 사운드 판별
                val rms = calcRms(value)
                val passed = rms > 500.0

                onDebugUpdate?.invoke(currentDebug.copy(micRms = rms.toInt(), micSent = passed))
                Log.d("BLE_MIC", "RMS=${"%.0f".format(rms)} ${if (passed) "→ 전송 버퍼 편입" else "→ 무음 폐기"}")

                if (passed) accumulateMicData(value)
            }
            VIB_CHAR_UUID -> {
                Log.w("BLE_VIB", "⚡ 충격 감지 트리거 수신")
                currentDebug = currentDebug.copy(lastEvent = "⚡ 충격 감지")
                onDebugUpdate?.invoke(currentDebug)
                onShockDetected?.invoke()
            }
        }
    }

    private var currentDebug = DebugInfo()

    /** PCM 데이터 RMS(Root Mean Square) 계산 유틸리티 */
    private fun calcRms(pcm: ByteArray): Double {
        if (pcm.size < 2) return 0.0
        val samples = ShortArray(pcm.size / 2)
        java.nio.ByteBuffer.wrap(pcm).order(java.nio.ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(samples)
        return Math.sqrt(samples.map { it.toDouble() * it }.average())
    }

    /** 오디오 청크 버퍼링 및 STT 변환용 WAV 헤더 병합 */
    private fun accumulateMicData(chunk: ByteArray) {
        synchronized(micBufferLock) {
            micBuffer.write(chunk)
            // 설정된 분량(3초) 도달 시 WAV 패키징 후 콜백 전송
            if (micBuffer.size() >= MIC_BUFFER_MAX) {
                val pcm = micBuffer.toByteArray()
                micBuffer.reset()

                val wav = buildWavBytes(pcm)
                currentDebug = currentDebug.copy(lastEvent = "🗣️ STT 전송")
                onDebugUpdate?.invoke(currentDebug)

                Log.d("STT", "WAV 파일 생성 및 전송 완료: ${wav.size} bytes")
                onSttDataReady?.invoke(wav)
            }
        }
    }

    /** RAW PCM 데이터 기반 WAV 규격 파일 헤더 생성기 */
    private fun buildWavBytes(pcm: ByteArray): ByteArray {
        val sr = 16000; val ch = 1; val bps = 16
        val buf = java.io.ByteArrayOutputStream()
        val h = java.nio.ByteBuffer.allocate(44).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        h.put("RIFF".toByteArray()); h.putInt(36 + pcm.size)
        h.put("WAVE".toByteArray()); h.put("fmt ".toByteArray())
        h.putInt(16); h.putShort(1); h.putShort(ch.toShort())
        h.putInt(sr); h.putInt(sr * ch * bps / 8)
        h.putShort((ch * bps / 8).toShort()); h.putShort(bps.toShort())
        h.put("data".toByteArray()); h.putInt(pcm.size)
        buf.write(h.array()); buf.write(pcm)
        return buf.toByteArray()
    }

    /** BLE 기기 측 Notification 구독 데스크립터 활성화 */
    @SuppressLint("MissingPermission")
    private fun enableNotify(gatt: BluetoothGatt, char: BluetoothGattCharacteristic, tag: String) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            val descriptor = char.getDescriptor(NOTIFY_DESCRIPTOR)
            if (descriptor != null) {
                gatt.writeDescriptor(descriptor, BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE)
                Log.d("BLE", "$tag Notify 구독 완료 (API 33+)")
            }
        } else {
            @Suppress("DEPRECATION")
            char.getDescriptor(NOTIFY_DESCRIPTOR)?.let {
                it.value = BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
                @Suppress("DEPRECATION")
                gatt.writeDescriptor(it)
                Log.d("BLE", "$tag Notify 구독 완료 (Legacy)")
            }
        }
    }

    /** DHT22 온습도 텍스트 규격 파싱 및 임계값 판독 */
    private fun parseDht22AndNotify(raw: String) {
        val parts = raw.split(",")
        if (parts.size < 2) return
        val temp  = parts[0].toFloatOrNull() ?: return
        val humid = parts[1].toFloatOrNull() ?: return

        // 온습도 결합 조건 기반 열사병 위험도 판정
        val isAlert = temp >= TEMP_ALERT_THRESHOLD && humid >= HUMID_ALERT_THRESHOLD

        onIoTDataUpdate?.invoke(temp, humid, isAlert)
        currentDebug = currentDebug.copy(temp = temp, humid = humid)
        onDebugUpdate?.invoke(currentDebug)
        Log.d("BLE", "DHT22: ${temp}°C, ${humid}%, 경고=$isAlert")
    }

    @SuppressLint("MissingPermission")
    fun disconnectBle() {
        bluetoothGatt?.disconnect()
        bluetoothGatt?.close()
        bluetoothGatt  = null
        isBleConnected = false
    }

    /** MAC 주소 기반 스캔 생략 다이렉트 연결 (기존 페어링 기록 활용) */
    @SuppressLint("MissingPermission")
    fun tryDirectConnect() {
        val prefs = getSharedPreferences("ble_prefs", Context.MODE_PRIVATE)
        val savedMac = prefs.getString("pico_mac", null)

        if (savedMac != null) {
            Log.d("BLE", "로컬 MAC 주소 기반 다이렉트 연결 시도: $savedMac")
            promoteForegroundToBle()

            val manager = getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
            val adapter = manager.adapter
            val device = adapter.getRemoteDevice(savedMac)

            // 기기 재부팅 대기 및 백그라운드 편입 자동 연결 허용
            device.connectGatt(
                this@MonitoringService,
                true,
                gattCallback,
                BluetoothDevice.TRANSPORT_LE
            )
        } else {
            Log.d("BLE", "사전 저장 MAC 부재. 초기 스캔 로직 가동.")
            startBleScan()
        }
    }
}

/* ================================================================
 * 디버그 오버레이 패널 렌더링용 데이터 모델
 * ================================================================ */
data class DebugInfo(
    val bleConnected: Boolean = false,
    val micRms:       Int     = 0,
    val micSent:      Boolean = false,
    val temp:         Float   = 0f,
    val humid:        Float   = 0f,
    val iotFiltered:  Boolean = false,  // 상태 미변화 트래픽 생략 판정
    val locationSec:  Int     = 0,      // 위치 주기 전송까지의 잔여 시간(초)
    val lastEvent:    String  = "-"
)