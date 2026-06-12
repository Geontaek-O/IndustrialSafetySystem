package com.example.ConSAFE

import android.content.Context
import android.util.Log
import androidx.health.connect.client.HealthConnectClient
import androidx.health.connect.client.permission.HealthPermission
import androidx.health.connect.client.records.HeartRateRecord
import androidx.health.connect.client.request.ReadRecordsRequest
import androidx.health.connect.client.time.TimeRangeFilter
import java.time.Instant
import java.time.temporal.ChronoUnit

/**
 * [HeartRateManager]
 * Health Connect API 연동 및 심박수(BPM) 데이터 수집, 분석 매니저.
 * - 갤럭시 워치 등 웨어러블 디바이스에서 측정된 심박수 호출 용도.
 * 현재 한계점 존재
 * - 웨어러블 기기가 구형(Tizen 사용)인 한계로 인해 심박수 실시간 측정 불가.
 * - 삼성 헬스 -> Health Connect -> 본 앱으로 데이터가 거쳐 전송되므로 변경 반영 속도 느림.
 */
class HeartRateManager(private val context: Context) {

    /**
     * Health Connect 클라이언트 인스턴스.
     * - 미설치 기기 크래시 방지를 위해 lazy 호출 및 가용성(isAvailable) 검사 후 생성 권장.
     */
    private val client by lazy { HealthConnectClient.getOrCreate(context) }

    /**
     * 앱 실행에 필요한 Health Connect 권한 목록.
     * - 심박수 읽기 권한(ReadPermission) 단일 요구.
     */
    val permissions = setOf(
        HealthPermission.getReadPermission(HeartRateRecord::class)
    )

    /**
     * 필수 권한 승인 여부 확인.
     * @return 권한 모두 승인 시 true, 미승인/오류 시 false
     */
    suspend fun hasPermissions(): Boolean {
        if (!isAvailable()) return false
        return try {
            val granted = client.permissionController.getGrantedPermissions()
            granted.containsAll(permissions)
        } catch (e: Exception) {
            Log.e(TAG, "권한 확인 오류: ${e.message}")
            false
        }
    }

    /**
     * 기기 내 Health Connect SDK 지원 및 설치 여부 확인.
     * @return 사용 가능 시 true
     */
    fun isAvailable(): Boolean {
        val status = HealthConnectClient.getSdkStatus(context)
        return status == HealthConnectClient.SDK_AVAILABLE
    }

    /**
     * 최근 N분간의 심박수 레코드 조회 및 위험도 분석.
     *
     * @param minutesAgo 조회할 과거 시간 범위 (분 단위, 기본 30분)
     * @return [HeartRateResult] (성공 시 데이터, 실패/데이터 없음 시 예외 상태)
     */
    suspend fun readRecentHeartRate(minutesAgo: Long = 30): HeartRateResult {
        // 클라이언트 초기화 전 기기 지원 여부 선행 방어
        if (!isAvailable()) {
            Log.w(TAG, "Health Connect 사용 불가 기기")
            return HeartRateResult.Error("Health Connect를 사용할 수 없는 기기입니다")
        }
        return try {
            // 권한 누락 시 크래시 방어
            val granted = client.permissionController.getGrantedPermissions()
            Log.d(TAG, "허용된 권한: $granted")
            Log.d(TAG, "필요 권한: $permissions")
            if (!granted.containsAll(permissions)) {
                Log.w(TAG, "❌ 권한 부족")
                return HeartRateResult.Error("권한이 허용되지 않았습니다")
            }

            // 조회 시간 범위 설정 (현재 시각 ~ N분 전)
            val endTime   = Instant.now()
            val startTime = endTime.minus(minutesAgo, ChronoUnit.MINUTES)
            Log.d(TAG, "조회 범위: $startTime ~ $endTime")

            // 레코드 쿼리 빌더
            val request = ReadRecordsRequest(
                recordType    = HeartRateRecord::class,
                timeRangeFilter = TimeRangeFilter.between(startTime, endTime)
            )

            // Health Connect 조회 수행
            val response = client.readRecords(request)
            val records  = response.records
            Log.d(TAG, "조회된 레코드 수: ${records.size}")

            if (records.isEmpty()) {
                Log.w(TAG, "⚠️ 최근 ${minutesAgo}분 내 데이터 없음 — 워치에서 수동 측정 필요")
                return HeartRateResult.NoData
            }

            // 최신 측정값 추출
            val latest    = records.last()
            val latestSample = latest.samples.lastOrNull()
            val latestBpm    = latestSample?.beatsPerMinute ?: 0L
            val latestTime   = latestSample?.time ?: Instant.now()

            // 구간 평균 심박수 산출
            val avgBpm       = records.flatMap { it.samples }
                .map { it.beatsPerMinute }
                .average()
                .toLong()

            Log.d(TAG, "✅ 최근 심박수: ${latestBpm}bpm, 평균: ${avgBpm}bpm, 측정시각: $latestTime")

            HeartRateResult.Success(
                latestBpm  = latestBpm,
                avgBpm     = avgBpm,
                alert      = classifyHeartRate(latestBpm),
                latestTime = latestTime
            )
        } catch (e: Exception) {
            Log.e(TAG, "심박수 읽기 오류: ${e.message}", e)
            HeartRateResult.Error(e.message ?: "알 수 없는 오류")
        }
    }

    /**
     * 현장 작업 기준 심박수 위험도 판별 로직.
     *
     * @param bpm 측정된 심박수
     * @return [HeartRateAlert] 상태 (정상/주의/빈맥/서맥 등)
     */
    private fun classifyHeartRate(bpm: Long): HeartRateAlert {
        return when {
            bpm <= 0   -> HeartRateAlert.UNKNOWN
            bpm < 50   -> HeartRateAlert.TOO_LOW      // 서맥 (작업자 이상 징후)
            bpm > 120  -> HeartRateAlert.TOO_HIGH     // 빈맥 (고위험 수준)
            bpm > 100  -> HeartRateAlert.HIGH         // 주의 (격렬한 활동 혹은 이상 징후)
            else       -> HeartRateAlert.NORMAL       // 정상 범위 (50~100)
        }
    }

    companion object {
        const val TAG = "HeartRate"
    }
}

/**
 * 심박수 쿼리 결과
 */
sealed class HeartRateResult {
    /** * 성공 상태 반환 모델
     * @param latestBpm 최신 심박수
     * @param avgBpm 구간 평균 심박수
     * @param alert 위험도 판별 결과
     * @param latestTime 마지막 측정 시점
     */
    data class Success(
        val latestBpm:  Long,
        val avgBpm:     Long,
        val alert:      HeartRateAlert,
        val latestTime: Instant
    ) : HeartRateResult()

    /** 데이터 부재 (측정 기록 없음) */
    object NoData                    : HeartRateResult()

    /** 쿼리 또는 권한 오류 발생 */
    data class Error(val msg: String): HeartRateResult()
}

/**
 * 심박수 위험도 분류 및 UI 매핑 열거형.
 * - 건설 현장 기준에 맞춘 임계값 및 직관적 컬러 코드 포함.
 */
enum class HeartRateAlert(val label: String, val color: Int) {
    NORMAL   ("정상",         android.graphics.Color.rgb(76, 175, 80)),   // 초록: 안정 상태
    HIGH     ("주의 (높음)",  android.graphics.Color.rgb(255, 152, 0)),   // 주황: 경계 요망
    TOO_HIGH ("위험 (빈맥)",  android.graphics.Color.rgb(244, 67, 54)),   // 빨강: 즉각 조치 필요
    TOO_LOW  ("위험 (서맥)",  android.graphics.Color.rgb(33, 150, 243)), // 파랑: 저혈압/쇼크 의심
    UNKNOWN  ("측정 불가",    android.graphics.Color.GRAY)              // 회색: 에러 상태
}