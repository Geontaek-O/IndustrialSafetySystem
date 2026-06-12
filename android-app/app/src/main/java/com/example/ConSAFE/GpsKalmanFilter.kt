package com.example.ConSAFE

/**
 * [GpsKalmanFilter]
 * 실시간 GPS 노이즈 스무딩 및 튐 현상 방지용 1D 칼만 필터.
 * - 센서 이상값(NaN, 음수 정확도, 시계 역행 등) 방어 로직 포함.
 */
class GpsKalmanFilter {
    private var lat = 0.0
    private var lon = 0.0
    private var variance = -1f
    private var initialized = false

    companion object {
        /**
         * GPS 측정 정확도(m) 최소 허용치.
         * 정확도 0.0 이하 보고 시 0 나눗셈(Divide by Zero) 오류 방지용.
         */
        private const val MIN_ACCURACY = 1f
    }

    /**
     * 신규 GPS 좌표 필터링 및 보정 결과 반환.
     *
     * @param newLat 신규 측정 위도
     * @param newLon 신규 측정 경도
     * @param accuracy 위치 측정 정확도 (m 단위)
     * @param isMoving 작업자 이동 여부 (true: 이동, false: 정지). 시스템 노이즈(Q) 동적 조정용.
     * @param elapsedMs 직전 업데이트 대비 경과 시간 (ms)
     * @return 보정된 (위도, 경도)
     */
    fun process(
        newLat: Double, newLon: Double, accuracy: Float,
        isMoving: Boolean = false, elapsedMs: Long = 1000L
    ): Pair<Double, Double> {

        // 1. 비정상 입력 1차 방어 (상태 오염 방지)
        // NaN 또는 Infinite 유입 시 로직 무시
        if (newLat.isNaN() || newLon.isNaN() ||
            newLat.isInfinite() || newLon.isInfinite()
        ) {
            // 초기화 전: (0.0, 0.0) 유지 / 초기화 후: 마지막 정상 좌표 반환
            return Pair(lat, lon)
        }

        // 2. 비정상 입력 2차 방어 (연산 오류 및 시계 역행 방지)
        // 정확도 0 이하 시 최소값 적용
        val safeAccuracy = if (accuracy.isNaN() || accuracy <= 0f) MIN_ACCURACY
        else accuracy

        // 경과 시간 음수 시 0 강제
        val safeElapsed = elapsedMs.coerceAtLeast(0L)

        if (!initialized || variance < 0) {
            // 최초 데이터 수신 시 초기화
            lat = newLat
            lon = newLon
            variance = safeAccuracy * safeAccuracy
            initialized = true
        } else {
            // 3. 상태 예측 (Predict)
            // 이동 시 Q=5 (신규 좌표 신뢰), 정지 시 Q=1 (기존 위치 유지)
            val q = if (isMoving) 5f else 1f
            variance += q * (safeElapsed / 1000f)

            // 4. 상태 업데이트 (Update)
            // 칼만 이득(Gain) 계산 및 위치, 오차 공분산 보정
            val gain = variance / (variance + safeAccuracy * safeAccuracy)
            lat += gain * (newLat - lat)
            lon += gain * (newLon - lon)

            variance *= (1 - gain)
        }

        return Pair(lat, lon)
    }

    /**
     * 필터 상태 초기화.
     * GPS 장기 소실 후 재연결 또는 구역 완전 이탈 시 오염 리셋 용도.
     */
    fun reset() {
        variance = -1f
        initialized = false
    }
}