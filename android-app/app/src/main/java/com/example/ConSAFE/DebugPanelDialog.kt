package com.example.ConSAFE

import android.app.Activity
import android.app.AlertDialog
import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import android.widget.ScrollView
import android.widget.TextView

/**
 * 디버그 정보 다이얼로그.
 * 데모 모드 활성 시 🛠️ 버튼으로 열림.
 * show() 후 update()를 반복 호출하면 내용이 실시간 갱신됨.
 */
class DebugPanelDialog(private val context: Context) {

    private var infoText: TextView? = null
    private var dialog: AlertDialog? = null

    fun show() {
        if (dialog?.isShowing == true) return
        if (!canShow()) return   // Activity 종료/파괴 중이면 표시하지 않음 (BadTokenException 방지)

        val tv = TextView(context).apply {
            textSize  = 12f
            typeface  = Typeface.MONOSPACE
            setTextColor(Color.WHITE)
            setBackgroundColor(0xFF111111.toInt())
            setPadding(dp(16), dp(12), dp(16), dp(12))
            text = "⏳ 데이터 수신 대기 중..."
        }
        infoText = tv

        val scroll = ScrollView(context).apply { addView(tv) }

        dialog = AlertDialog.Builder(context)
            .setTitle("🛠️ 디버그 패널")
            .setView(scroll)
            .setNegativeButton("닫기") { d, _ ->
                d.dismiss()
                infoText = null
            }
            .create()

        try {
            dialog?.show()
        } catch (e: Exception) {
            // 표시 직전 Activity가 사라진 경우 등
            dialog = null
            infoText = null
        }
    }

    /** Activity 컨텍스트가 살아있어 다이얼로그를 띄울 수 있는지 확인 */
    private fun canShow(): Boolean {
        val activity = context as? Activity ?: return true  // Activity가 아니면 판단 불가 → 시도 허용
        return !activity.isFinishing && !activity.isDestroyed
    }

    /** MonitoringService의 onDebugUpdate 콜백에서 호출 */
    fun update(info: DebugInfo, lastSentTime: Long) {
        val tv = infoText ?: return
        if (dialog?.isShowing != true) return

        val bleIcon  = if (info.bleConnected) "🔵" else "🔴"
        val micIcon  = if (info.micSent)      "✅ 전송됨" else "🔇 무음 생략"
        val locSec   = ((System.currentTimeMillis() - lastSentTime) / 1000).coerceAtLeast(0)
        val nextLoc  = maxOf(0, 30 - locSec.toInt())
        val bgColor  = if (info.bleConnected) 0xFF003300.toInt() else 0xFF1A0000.toInt()

        tv.setBackgroundColor(bgColor)
        tv.text = buildString {
            appendLine("┌─────────────────────────┐")
            appendLine("│        디버그 정보        │")
            appendLine("└─────────────────────────┘")
            appendLine()
            appendLine("$bleIcon  BLE 연결")
            appendLine("   → ${if (info.bleConnected) "연결됨" else "미연결 (재스캔 대기)"}")
            appendLine()
            appendLine("🎤  마이크 RMS")
            appendLine("   → ${info.micRms}   (임계값: 500)")
            appendLine("   → $micIcon")
            appendLine()
            appendLine("📍  위치 전송")
            appendLine("   → ${nextLoc}초 후 전송 (정지 기준 30초)")
            appendLine()
            appendLine("🌡️  온습도")
            appendLine("   → ${info.temp}°C  /  ${info.humid}%")
            appendLine()
            appendLine("📌  마지막 이벤트")
            append  ("   → ${info.lastEvent}")
        }
    }

    fun isShowing() = dialog?.isShowing == true

    private fun dp(value: Int): Int =
        (value * context.resources.displayMetrics.density).toInt()
}