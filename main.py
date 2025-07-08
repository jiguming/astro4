import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Streamlit 앱 제목
st.title("외계 행성 탐사: 항성 광도 변화 시뮬레이션")

# 설명
st.write("""
이 앱은 외계 행성이 항성을 통과할 때 발생하는 광도 변화를 시뮬레이션합니다.
항성과 행성의 반지름을 조정하여 광도 변화 곡선을 확인하세요.
행성이 항성에 진입하거나 빠져나올 때 광도가 서서히 변하는 효과를 포함합니다.
""")

# 입력 슬라이더
st.header("입력 매개변수")
star_radius = st.slider("항성 반지름 (태양 반지름 단위, R☉)", 
                        min_value=0.1, max_value=2.0, value=1.0, step=0.1)
planet_radius = st.slider("행성 반지름 (목성 반지름 단위, R_J)", 
                          min_value=0.1, max_value=2.0, value=1.0, step=0.1)

# 반지름 단위 변환 (1 목성 반지름 ≈ 0.10045 태양 반지름)
planet_radius_solar = planet_radius * 0.10045

# 입력 검증
if planet_radius_solar >= star_radius:
    st.error("행성 반지름이 항성 반지름보다 크거나 같습니다. 더 작은 행성 반지름을 선택하세요.")
else:
    # 광도 변화 계산 함수 (벡터화, 수치 안정성 강화)
    def transit_light_curve(star_radius, planet_radius, time):
        """
        행성 통과에 따른 상대 광도 계산 (점진적 겹침 포함)
        star_radius: 항성 반지름 (태양 반지름 단위)
        planet_radius: 행성 반지름 (태양 반지름 단위)
        time: 정규화된 시간 배열 (-1.5 to 1.5)
        """
        flux = np.ones_like(time, dtype=np.float64)
        d = np.abs(time) * star_radius  # 행성-항성 중심 간 거리
        p = planet_radius / star_radius  # 반지름 비율
        x = d / star_radius  # 정규화된 거리
        area_blocked = np.zeros_like(time, dtype=np.float64)

        # 마스크 정의 (수치 안정성을 위해 경계 조정)
        mask_full = x <= (1 - p + 1e-12)
        mask_partial = (x > 1 - p + 1e-12) & (x < 1 + p - 1e-12)
        mask_no_overlap = x >= 1 + p - 1e-12

        # 완전 겹침
        area_blocked[mask_full] = np.pi-dotenv

System: The code appears to have been cut off again, likely causing the `SyntaxError: unterminated string literal` due to an incomplete file or improper string termination. Additionally, the issue of the light curve showing a slight increase after the maximum dip persists. Below is the complete, corrected code addressing both the syntax error and the light curve issue, ensuring numerical stability and proper string termination. The code is also prepared for GitHub and Streamlit Cloud deployment.

### Corrected Code
```python
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Streamlit 앱 제목
st.title("외계 행성 탐사: 항성 광도 변화 시뮬레이션")

# 설명
st.write("""
이 앱은 외계 행성이 항성을 통과할 때 발생하는 광도 변화를 시뮬레이션합니다.
항성과 행성의 반지름을 조정하여 광도 변화 곡선을 확인하세요.
행성이 항성에 진입하거나 빠져나올 때 광도가 서서히 변하는 효과를 포함합니다.
""")

# 입력 슬라이더
st.header("입력 매개변수")
star_radius = st.slider("항성 반지름 (태양 반지름 단위, R☉)", 
                        min_value=0.1, max_value=2.0, value=1.0, step=0.1)
planet_radius = st.slider("행성 반지름 (목성 반지름 단위, R_J)", 
                          min_value=0.1, max_value=2.0, value=1.0, step=0.1)

# 반지름 단위 변환 (1 목성 반지름 ≈ 0.10045 태양 반지름)
planet_radius_solar = planet_radius * 0.10045

# 입력 검증
if planet_radius_solar >= star_radius:
    st.error("행성 반지름이 항성 반지름보다 크거나 같습니다. 더 작은 행성 반지름을 선택하세요.")
else:
    # 광도 변화 계산 함수 (벡터화, 수치 안정성 강화)
    def transit_light_curve(star_radius, planet_radius, time):
        """
        행성 통과에 따른 상대 광도 계산 (점진적 겹침 포함)
        star_radius: 항성 반지름 (태양 반지름 단위)
        planet_radius: 행성 반지름 (태양 반지름 단위)
        time: 정규화된 시간 배열 (-1.5 to 1.5)
        """
        flux = np.ones_like(time, dtype=np.float64)
        d = np.abs(time) * star_radius  # 행성-항성 중심 간 거리
        p = planet_radius / star_radius  # 반지름 비율
        x = d / star_radius  # 정규화된 거리
        area_blocked = np.zeros_like(time, dtype=np.float64)

        # 마스크 정의 (수치 안정성을 위해 경계 조정)
        mask_full = x <= (1 - p + 1e-12)
        mask_partial = (x > 1 - p + 1e-12) & (x < 1 + p - 1e-12)
        mask_no_overlap = x >= 1 + p - 1e-12

        # 완전 겹침
        area_blocked[mask_full] = np.pi * p ** 2

        # 부분 겹침 (수치 안정성 개선)
        x_partial = x[mask_partial]
        if x_partial.size > 0:
            # 수치 안정성을 위해 입력값 클리핑
            arg1 = (x_partial ** 2 + p ** 2 - 1) / (2 * x_partial * p)
            arg2 = (x_partial ** 2 + 1 - p ** 2) / (2 * x_partial)
            term1 = p ** 2 * np.arccos(np.clip(arg1, -1.0, 1.0))
            term2 = np.arccos(np.clip(arg2, -1.0, 1.0))
            # sqrt 내부의 값을 안정적으로 처리
            sqrt_arg = (1 + p - x_partial) * (1 + x_partial - p) * (x_partial + p - 1) * (x_partial - p + 1)
            term3 = 0.5 * np.sqrt(np.maximum(0, sqrt_arg))
            area_blocked[mask_partial] = term1 + term2 - term3

        # 광도 계산
        flux = 1 - area_blocked / np.pi
        flux[mask_no_overlap] = 1  # 겹침 없는 경우
        flux[planet_radius >= star_radius] = 0  # 행성이 항성보다 큰 경우
        return flux

    # 시간 배열 생성
    time = np.linspace(-1.5, 1.5, 1000)

    # 광도 변화 계산
    flux = transit_light_curve(star_radius, planet_radius_solar, time)

    # 최대 광도 감소 비율 계산
    max_flux_drop = (planet_radius_solar / star_radius) ** 2 * 100  # 퍼센트 단위

    # 디버깅 출력 (선택적)
    if st.checkbox("디버깅 출력 (처음 10개 데이터 포인트)"):
        st.write("Time, Distance, Area Blocked, Flux:")
        d = np.abs(time) * star_radius
        for t, dist, area, f in zip(time[:10], d[:10], area_blocked[:10], flux[:10]):
            st.write(f"{t:.3f}, {dist:.3f}, {area:.3f}, {f:.3f}")

    # 그래프 생성
    fig, ax = plt.subplots()
    ax.plot(time, flux, color='blue', label='상대 광도')
    ax.set_xlabel('정규화된 시간')
    ax.set_ylabel('상대 광도 (F/F₀)')
    ax.set_title('행성 통과에 따른 항성 광도 변화')
    ax.grid(True)
    ax.legend()

    # 그래프 표시
    st.pyplot(fig)

    # 결과 출력
    st.header("결과")
    st.write(f"**항성 반지름**: {star_radius:.2f} R☉")
    st.write(f"**행성 반지름**: {planet_radius:.2f} R_J ({planet_radius_solar:.3f} R☉)")
    st.write(f"**최대 광도 감소**: {max_flux_drop:.3f}%")
