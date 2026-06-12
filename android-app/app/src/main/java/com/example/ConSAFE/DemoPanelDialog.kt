package com.example.ConSAFE

import android.app.Activity
import android.app.AlertDialog
import android.content.Context
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView

/**
 * 시연용 데모 패널 다이얼로그.
 * 심박수/위치/IoT 값을 강제 주입하여 시연 가능.
 */
class DemoPanelDialog(
    private val context: Context,
    private val callbacks: Callbacks
) {

    interface Callbacks {
        fun onHeartRateInject(bpm: Long)
        fun onLocationInject(lat: Double, lon: Double, label: String)
        fun onIoTInject(temp: Float, humid: Float, alert: Boolean)
    }

    fun show() {
        // Activity 종료/파괴 중이면 표시하지 않음 (BadTokenException 방지)
        val activity = context as? Activity
        if (activity != null && (activity.isFinishing || activity.isDestroyed)) return

        val container = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(16), dp(20), dp(16))
        }

        // 심박수 섹션
        container.addView(sectionTitle("심박수"))
        container.addView(rowButtons(listOf(
            "정상 75"      to { callbacks.onHeartRateInject(75L) },
            "주의 110"     to { callbacks.onHeartRateInject(110L) },
            "위험 빈맥 145" to { callbacks.onHeartRateInject(145L) },
            "위험 서맥 40" to { callbacks.onHeartRateInject(40L) }
        )))


        // IoT 섹션
        container.addView(sectionTitle("환경 (온습도)"))
        container.addView(rowButtons(listOf(
            "정상" to {
                val (t, h, a) = DemoController.FakeIoT.NORMAL
                callbacks.onIoTInject(t, h, a)
            },
            "🔥 열사병 위험" to {
                val (t, h, a) = DemoController.FakeIoT.HEAT_RISK
                callbacks.onIoTInject(t, h, a)
            }
        )))

        try {
            AlertDialog.Builder(context)
                .setTitle("시연 시뮬레이션 패널")
                .setView(container)
                .setNegativeButton("닫기", null)
                .show()
        } catch (e: Exception) {
            // 표시 직전 Activity가 사라진 경우 무시
        }
    }

    /* ================================================================
     * UI 헬퍼
     * ================================================================ */
    private fun sectionTitle(text: String): TextView = TextView(context).apply {
        this.text = text
        textSize  = 14f
        setTextColor(0xFF3F51B5.toInt())
        setTypeface(typeface, android.graphics.Typeface.BOLD)
        setPadding(0, dp(12), 0, dp(6))
    }

    private fun rowButtons(items: List<Pair<String, () -> Unit>>): LinearLayout {
        // 화면 좁으면 2열로 줄바꿈
        val outer = LinearLayout(context).apply { orientation = LinearLayout.VERTICAL }
        items.chunked(2).forEach { pair ->
            val row = LinearLayout(context).apply {
                orientation = LinearLayout.HORIZONTAL
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                )
            }
            pair.forEach { (label, action) ->
                val btn = Button(context).apply {
                    text = label
                    textSize = 11f
                    setOnClickListener { action() }
                    layoutParams = LinearLayout.LayoutParams(
                        0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f
                    ).apply { setMargins(dp(2), dp(2), dp(2), dp(2)) }
                }
                row.addView(btn)
            }
            outer.addView(row)
        }
        return outer
    }

    private fun dp(value: Int): Int =
        (value * context.resources.displayMetrics.density).toInt()
}