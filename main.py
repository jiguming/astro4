import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Streamlit 앱 제목
st.title("외계 행성 탐사: 항성 광도 변화 시뮬레이션 (림 다크닝 포함)")

# 설명
st.write("""
이 앱은 외계 행성이 항성을 통과할 때 발생하는 광도 변화를 시뮬레이션합니다.
항성과 행성의 반지름 및 림 다크닝 계수를 조정하여 광도 변화 곡선을 확인하세요.
림 다크닝은 항성 중심이 가장 밝고 가장자리로 갈수록 어두워지는 효과를 반영합니다.
""")

# 입력 슬라이더
st.header("입력 매개변수")
star_radius = st.slider("항성 반지름 (태양 반지름 단위, R☉)", 
                        min_value=0.1, max_value=2.0, value=1.0, step=0.1)
planet_radius = st.slider("행성 반지름 (목성 반지름 단위, R_J)", 
                          min_value=0.1, max_value=2.0, value=1.0, step=0.1)
limb_darkening_coeff = st.slider("림 다크닝 계수 (u)", 
                                min_value=0.0, max_value=1.0, value=0.5, step=0.05)

# 반지름 단위 변환 (1 목성 반지름 ≈ 0.10045 태양 반지름)
planet_radius_solar = planet_radius * 0.10045

# 광도 변화 계산 함수 (림 다크닝 포함)
def transit_light_curve_limb_darkening(star_radius, planet_radius, time, u):
    """
    행성 통과에 따른 상대 광도 계산 (림 다크닝 포함)
    star_radius: 항성 반지름 (태양 반지름 단위)
    planet_radius: 행성 반지름 (태양 반지름 단위)
    time: 정규화된 시간 배열 (-1.5 to 1.5)
    u: 선형 림 다크닝 계수
    """
    flux = np.ones_like(time)  # 기본 광도 = 1
    transit_mask = np.abs(time) <= 1  # 행성이 항성을 가리는 구간

    # 림 다크닝 없는 경우의 광도 (참고용)
    flux_no_limb = np.ones_like(time)
    flux_no_limb[transit_mask] = 1 - (planet_radius / star_radius) ** 2

    # 림 다크닝 포함 광도 계산
    for i, t in enumerate(time):
        if transit_mask[i]:
            # 행성 중심과 항성 중심 사이의 거리 (정규화된 시간에 비례)
            d = np.abs(t) * star_radius  # 정규화된 거리
            if d <= star_radius + planet_radius and d >= abs(star_radius - planet_radius):
                # 부분 겹침 또는 완전 겹침
                if planet_radius >= star_radius:
                    # 행성이 항성보다 큰 경우 (완전 가림)
                    flux[i] = 0
                else:
                    # 겹치는 면적 계산 (원형 가정)
                    p = planet_radius / star_radius
                    x = d / star_radius
                    if x <= 1 - p:  # 완전 겹침
                        area_blocked = np.pi * p ** 2
                    else:  # 부분 겹침
                        # 두 원의 교차 면적 계산 (기하학적 공식)
                        term1 = p ** 2 * np.arccos((x ** 2 + p ** 2 - 1) / (2 * x * p))
                        term2 = np.arccos((x ** 2 + 1 - p ** 2) / (2 * x))
                        term3 = 0.5 * np.sqrt((1 + p - x) * (1 + x - p) * (x + p - 1) * (x - p + 1))
                        area_blocked = term1 + term2 - term3

                    # 림 다크닝 적용: 항성 표면의 밝기 분포
                    # 가정: 균일 원반 대비 가려진 부분의 밝기 손실 계산
                    mu = np.sqrt(1 - (d / star_radius) ** 2) if d < star_radius else 0
                    intensity = 1 - u * (1 - mu)  # 선형 림 다크닝
                    flux[i] = 1 - area_blocked * intensity / np.pi

    return flux, flux_no_limb

# 시간 배열 생성 (정규화된 시간, -1.5 ~ 1.5)
time = np.linspace(-1.5, 1.5, 1000)

# 광도 변화 계산
flux, flux_no_limb = transit_light_curve_limb_darkening(star_radius, planet_radius_solar, time, limb_darkening_coeff)

# 최대 광도 감소 비율 계산 (림 다크닝 없는 경우로 단순화)
max_flux_drop = (planet_radius_solar / star_radius) ** 2 * 100  # 퍼센트 단위

# 그래프 생성
fig, ax = plt.subplots()
ax.plot(time, flux, color='blue', label='상대 광도 (림 다크닝 포함)')
ax.plot(time, flux_no_limb, color='red', linestyle='--', label='상대 광도 (림 다크닝 미포함)')
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
st.write(f"**림 다크닝 계수 (u)**: {limb_darkening_coeff:.2f}")
st.write(f"**최대 광도 감소 (림 다크닝 미포함 기준)**: {max_flux_drop:.3f}%")

# 추가 정보
st.write("""
### 참고
- **림 다크닝**: 항성 중심이 가장 밝고 가장자리로 갈수록 어두워지는 효과를 반영합니다. 선형 림 다크닝 법칙 \( I(\mu) = I_0 [1 - u (1 - \mu)] \)를 사용했습니다.
- 광도 감소는 행성이 항성을 가리는 면적과 림 다크닝에 따른 밝기 분포를 고려하여 계산됩니다.
- 시간은 정규화된 단위로, 실제 통과 시간은 궤도 주기와 항성 크기에 따라 달라집니다.
- 이 모델은 단순화를 위해 단일 선형 림 다크닝 계수를 사용하며, 더 복잡한 비선형 모델은 포함하지 않습니다.
""")
