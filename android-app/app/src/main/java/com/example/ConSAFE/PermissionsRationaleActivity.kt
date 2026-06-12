package com.example.ConSAFE

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

/**
 * Health Connect 권한 사용 근거 화면.
 * 사용자가 권한 요청 다이얼로그에서 "더 알아보기"를 누르면 이 화면이 표시됨.
 * Health Connect가 앱을 신뢰할 수 있는 앱으로 인식하기 위해 반드시 필요.
 */
class PermissionsRationaleActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 간단한 안내 텍스트
        val textView = TextView(this).apply {
            text = """
                심박수 데이터 수집 안내
                
                본 앱은 다음 목적으로 심박수 데이터를 사용합니다:
                
                • 작업자의 건강 상태 모니터링
                • 위험 수준의 심박수 감지 시 경고 알림
                • 긴급 상황 발생 시 관리자에게 자동 알림
                
                수집된 데이터는 개인 안전 관리 목적으로만 사용되며,
                제3자에게 제공되지 않습니다.
            """.trimIndent()
            textPaddingDp(24)
            textSize = 14f
            setLineSpacing(8f, 1f)
        }

        setContentView(textView)
    }

    private fun TextView.textPaddingDp(dp: Int) {
        val px = (dp * resources.displayMetrics.density).toInt()
        setPadding(px, px, px, px)
    }
}