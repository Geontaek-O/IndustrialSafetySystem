package com.example.ConSAFE

import android.content.Context

/**
 * 시연용 데모 모드 관리.
 * 로그인 버튼 1초간 누를 시 활성화.
 */
object DemoController {

    private const val PREFS  = "demo_prefs"
    private const val KEY_ON = "demo_mode"

    /* 데모 모드 활성 여부 */
    fun isEnabled(context: Context): Boolean {
        return context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .getBoolean(KEY_ON, false)
    }

    fun setEnabled(context: Context, enabled: Boolean) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit()
            .putBoolean(KEY_ON, enabled)
            .apply()
    }

    /* 심박수 시나리오 */
    object FakeHeartRates {
        const val NORMAL    = 75L   // 정상
        const val HIGH      = 110L  // 주의
        const val TOO_HIGH  = 145L  // 위험 (빈맥)
        const val TOO_LOW   = 40L   // 위험 (서맥)
    }

    /* IoT 환경 시나리오 */
    object FakeIoT {
        val NORMAL    = Triple(24.0f, 50.0f, false)  // 온도, 습도, 경고
        val HEAT_RISK = Triple(38.0f, 80.0f, true)   // 열사병 위험
    }
}