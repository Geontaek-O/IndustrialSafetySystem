package com.example.ConSAFE

import android.content.Intent
import android.os.Bundle
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen

/**
 * [LoginActivity]
 * 앱 진입점 및 사용자 인증 처리 액티비티.
 * - 주요 기능: 서버 로그인 연동, 자동 로그인(세션 유지), 데모/테스트 모드 진입 지원.
 */
class LoginActivity : AppCompatActivity() {

    // 싱글톤 API 인터페이스 인스턴스
    private val api = ApiService.create()

    override fun onCreate(savedInstanceState: Bundle?) {
        // Android 12+ Splash Screen API 적용
        installSplashScreen()
        super.onCreate(savedInstanceState)

        // 1. 자동 로그인 검사
        // - UI 렌더링(setContentView) 전 세션 유무를 파악하여 불필요한 화면 깜빡임 방지
        if (isAlreadyLoggedIn()) {
            navigateToMain()
            return
        }

        setContentView(R.layout.activity_login)

        // UI 컴포넌트 바인딩
        val etId        = findViewById<EditText>(R.id.etLoginEmail)
        val etPw        = findViewById<EditText>(R.id.etLoginPw)
        val btnLogin    = findViewById<Button>(R.id.btnLogin)
        val btnToSignup = findViewById<Button>(R.id.btnToSignup)

        // 2. 일반 로그인 버튼 이벤트
        btnLogin.setOnClickListener {
            val id = etId.text.toString().trim()
            val pw = etPw.text.toString()

            // 입력값 유효성 1차 검증
            if (id.isEmpty() || pw.isEmpty()) {
                Toast.makeText(this, "이메일과 비밀번호를 입력해주세요.", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            performLogin(id, pw)
        }

        // 3. 회원가입 화면 이동
        btnToSignup.setOnClickListener {
            startActivity(Intent(this, RegisterActivity::class.java))
        }

        // 4. 시연용 데모 모드 진입 (히든 액션)
        // - 버튼 3초 롱클릭 시 데모 상태 토글 처리
        btnLogin.setOnLongClickListener {
            val current = DemoController.isEnabled(this)
            DemoController.setEnabled(this, !current)
            val msg = if (!current) "🎬 데모 모드 ON" else "데모 모드 OFF"
            Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
            true   // 일반 클릭 이벤트(로그인 실행) 차단
        }
    }

    /**
     * 로컬 저장소 기반 기기 내 로그인 세션 유무 판별.
     * @return 세션(이메일) 존재 시 true
     */
    private fun isAlreadyLoggedIn(): Boolean {
        val prefs = getSharedPreferences("user_prefs", MODE_PRIVATE)
        val email = prefs.getString("user_email", "")
        return !email.isNullOrEmpty()
    }

    /**
     * 실제 로그인 로직 처리 (서버 통신 및 하드코딩 테스트 계정 우회).
     * @param id 입력된 이메일
     * @param pw 입력된 비밀번호
     */
    private fun performLogin(id: String, pw: String) {

        // [예외 케이스] 하드코딩된 오프라인 테스트 계정 우회 (서버 OFF 상태에서 테스트를 위함.)
        if (id == "test" && pw == "1234") {
            getSharedPreferences("user_prefs", MODE_PRIVATE)
                .edit()
                .putString("user_email", "test")
                .putString("user_name", "테스트")
                .putString("user_gender", "M")
                .putBoolean("is_admin", false)
                .putString("user_profile_url", "")   // 테스트 계정 프로필 이미지 미제공
                .apply()

            Toast.makeText(this, "테스트 계정으로 로그인했습니다.", Toast.LENGTH_SHORT).show()
            navigateToMain()
            return
        }

        // [정상 케이스] 서버 API 연동 로그인
        lifecycleScope.launch {
            try {
                val response = api.login(mapOf("email" to id, "password" to pw))

                if (response.isSuccessful && response.body()?.success == true) {
                    val body = response.body()!!

                    // 서버 응답 데이터를 로컬 SharedPreferences에 캐싱
                    getSharedPreferences("user_prefs", MODE_PRIVATE)
                        .edit()
                        .putString("user_email",       id)
                        .putString("user_name",        body.name)
                        .putString("user_gender",      body.gender)
                        .putBoolean("is_admin",        body.isAdmin)
                        .putString("user_profile_url", body.profile_url ?: "")
                        .apply()

                    Toast.makeText(this@LoginActivity, "${body.name}님 환영합니다!", Toast.LENGTH_SHORT).show()
                    navigateToMain()
                } else {
                    // 서버 통신은 성공했으나 인증 실패 (비밀번호 불일치 등)
                    val msg = response.errorBody()?.string() ?: "로그인 실패"
                    Toast.makeText(this@LoginActivity, msg, Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                // 네트워크 오류, 타임아웃 등 통신 자체 실패
                Toast.makeText(this@LoginActivity, "서버 연결 실패: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }
    }

    /**
     * 메인 화면으로 이동 및 현재 로그인 화면 스택 제거.
     */
    private fun navigateToMain() {
        startActivity(Intent(this, MainActivity::class.java))
        finish()
    }
}