package com.example.ConSAFE

import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.RequestBody
import retrofit2.Response
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.*
import java.util.concurrent.TimeUnit

// ================================================================
// 서버 응답 및 요청 데이터 모델
// ================================================================

/**
 * 회원가입 결과 응답
 */
data class RegisterResponse(val success: Boolean, val message: String)

/**
 * 로그인 결과 및 사용자 기본 정보 응답
 */
data class LoginResponse(
    val success:     Boolean,
    val name:        String,
    val gender:      String,
    val isAdmin:     Boolean,
    val profile_url: String? = null   // 서버에 저장된 사용자 프로필 이미지의 상대 경로 (예: /static/faces/xxx.jpg)
)

/**
 * 작업자의 실시간 위치 정보 전송 요청
 */
data class SendLocationRequest(
    val email:       String,
    val lat:         Double,
    val lon:         Double,
    val sensor_type: String = "GPS"   // 추후 PDR 등 복합 측위 센서 타입 확장을 고려한 필드
)

/**
 * 위치, 심박수, IoT 등 일반적인 상태 전송에 대한 범용 응답
 * 작업자가 위험구역에 진입했을 경우 zone_alert 객체를 통해 경고 메시지를 수신함
 */
data class SendLocationResponse(
    val success:    Boolean,
    val message:    String,
    val zone_alert: ZoneAlert? = null   // null인 경우 정상 구역, 값이 존재하면 즉각적인 위험구역 경고 처리 필요
)

/**
 * 현장 구역(Zone) 기반의 안전 알림 상세 정보
 */
data class ZoneAlert(
    val zone_name: String,   // 감지된 구역 명칭
    val zone_type: String,   // 구역 분류
    val message:   String    // 사용자에게 노출할 알림 메시지
)

/**
 * 웨어러블 디바이스에서 측정된 심박수 데이터 전송 요청
 */
data class HeartRateRequest(
    val email:   String,
    val bpm:     Long,       // 현재 심박수
    val avg_bpm: Long,       // 구간 평균 심박수
    val status:  String      // 1차 판별 상태 ("NORMAL", "HIGH", "TOO_HIGH(빈맥)", "TOO_LOW(서맥)")
)

/**
 * 센서 보드에서 수집된 현장 주변 온습도 데이터 전송 요청
 */
data class IoTDataRequest(
    val email:       String,
    val temperature: Float,
    val humidity:    Float,
    val heat_alert:  Boolean // 온열 질환 위험 수준 도달 여부
)

/**
 * 충격 감지 및 음성 구조 요청(SOS)을 통합 처리하는 긴급 상황 알림 요청
 */
data class SosRequest(
    val email:          String,
    val emergency_type: String,   // 발생 요인 분류 ("IMPACT": 물리적 충격 감지 / "VOICE_SOS": 작업자 음성 요청)
    val message:        String    // 서버 및 관리자에게 전달할 긴급 상황 상세 내용
)

/**
 * 음성 기반 구조 요청(WAV) 전송에 대한 AI STT 분석 결과 응답
 */
data class SttAlertResponse(
    val success:  Boolean,
    val is_sos:   Boolean,   // AI 분석 결과 실제 구조 요청으로 판별되었는지 여부 (true 시 긴급 프로토콜 가동)
    val message:  String     // STT 변환 텍스트 또는 처리 결과 메시지
)

// ================================================================
// API 엔드포인트 인터페이스
// ================================================================
interface ApiService {

    /**
     * 신규 작업자 등록 (초기 안면 인식 데이터 확보를 위한 영상 파일 포함)
     */
    @Multipart
    @POST("register")
    suspend fun register(
        @Part("name")      name:      RequestBody,
        @Part("gender")    gender:    RequestBody,
        @Part("email")     email:     RequestBody,
        @Part("password")  password:  RequestBody,
        @Part("phone")     phone:     RequestBody,
        @Part("birthdate") birthdate: RequestBody,
        @Part             video:     MultipartBody.Part, // 안면 데이터 추출용 짧은 영상
    ): Response<RegisterResponse>

    @POST("api/login")
    suspend fun login(
        @Body body: Map<String, String>
    ): Response<LoginResponse>

    /**
     * 주기적인 작업자 위치 업데이트 및 위험구역 진입 여부 체크
     */
    @POST("api/send_location")
    suspend fun sendLocation(
        @Body location: SendLocationRequest
    ): Response<SendLocationResponse>

    /**
     * 로그아웃 처리 및 서버 상의 활성 작업자 위치 모니터링 해제
     */
    @POST("api/logout")
    suspend fun logout(
        @Body body: Map<String, String>
    ): Response<SendLocationResponse>

    @POST("api/send_heart_rate")
    suspend fun sendHeartRate(
        @Body data: HeartRateRequest
    ): Response<SendLocationResponse>

    @POST("api/send_iot_data")
    suspend fun sendIoTData(
        @Body data: IoTDataRequest
    ): Response<SendLocationResponse>

    /**
     * 긴급 상황 발생 시(충격 또는 수동 SOS) 즉각적인 알림 전송
     */
    @POST("api/emergency_sos")
    suspend fun sendEmergencySos(
        @Body request: SosRequest
    ): Response<SendLocationResponse>

    /**
     * 음성 SOS 요청 전송 (WAV 오디오 파일과 당시 위치 정보를 동시 전송하여 AI STT 검증 수행)
     */
    @Multipart
    @POST("api/stt_alert")
    suspend fun sendSttAlert(
        @Part("email") email: RequestBody,
        @Part("lat")   lat:   RequestBody,
        @Part("lon")   lon:   RequestBody,
        @Part          audio: MultipartBody.Part   // 분석을 위한 WAV 바이너리 데이터
    ): Response<SttAlertResponse>

    companion object {
        // ngrok 서버 주소
        const val BASE_URL = "https://toya-parapodial-faultlessly.ngrok-free.dev/"

        /**
         * 앱 전역에서 공유되는 단일 OkHttpClient 인스턴스.
         */
        private val okHttpClient: OkHttpClient by lazy {
            OkHttpClient.Builder()
                .addInterceptor { chain ->
                    // ngrok 무료 버전 사용 시 발생하는 브라우저 경고 페이지를 우회하기 위한 헤더
                    val request = chain.request().newBuilder()
                        .addHeader("ngrok-skip-browser-warning", "true")
                        .build()
                    chain.proceed(request)
                }
                .connectTimeout(30,  TimeUnit.SECONDS)
                .writeTimeout(120,   TimeUnit.SECONDS)   // 영상 파일 및 WAV 업로드 대기 시간 확보
                .readTimeout(180,    TimeUnit.SECONDS)   // 서버 측의 AI 기반 안면 추출 및 STT 처리 대기
                .callTimeout(240,    TimeUnit.SECONDS)   // 단일 요청의 전체 생명주기 상한선
                .build()
        }

        /**
         * ApiService 싱글톤 프록시 객체
         */
        private val instance: ApiService by lazy {
            Retrofit.Builder()
                .baseUrl(BASE_URL)
                .client(okHttpClient)
                .addConverterFactory(GsonConverterFactory.create())
                .build()
                .create(ApiService::class.java)
        }

        fun create(): ApiService = instance
    }
}