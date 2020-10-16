### Risky Asset + Risk-free Asset 시계열 모멘텀 전략 벡테스트
# Monthly rebalancing 가정
# tau : 모멘텀 측정 기간/ stDateNum : 투자시작일 / 이 둘을 바꿔가며 back-test

# 디렉토리 확인
import os
os.getcwd()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import FinanceDataReader as fdr


# =====================================================
# Data 로딩 및 변수 정의
# =====================================================
# 날짜 기준으로 오름차순 정렬된 Data 불러오기

data_excel = pd.ExcelFile('C:/Users/b_jun/트레이딩/HW09_김범준.xlsx')
data_df = data_excel.parse(sheet_name = 'data_m')
rawTime = data_df.iloc[1:, 0].copy().reset_index(drop=True)
rawRisky = data_df.iloc[1:, 1].copy().reset_index(drop=True)
rawRf = data_df.iloc[1:, 3].copy().reset_index(drop=True)

# 전체 기간에 대한 HPR(단위 : %)
rawR1 = rawRisky.pct_change()*100
rawR2 = rawRf.shift(1)/12

# 전체 기간에 대한 Risky asset 의 모멘텀 
tau = 12 # 모멘텀의 측정기간(월)
rawMom = rawRisky.pct_change(tau)*100 
rawSignal = 1*(rawMom > 0) # 모멘텀 있으면 1, 없으면 0

# =====================================================
#### Back Test 시작시점을 기준으로 데이터 정리
# =====================================================

stDateNum =  19941228  #20201230  # 투자시작일
stDate = pd.to_datetime(str(stDateNum), format= '%Y%m%d')
idx = np.argmin(np.abs(rawTime - stDate)) 

Time = rawTime[idx:].copy().reset_index(drop=True) # 깊은 복사
R1 = rawR1[idx:].copy().reset_index(drop=True)
R2 = rawR2[idx:].copy().reset_index(drop=True)
Signal = rawSignal.iloc[idx:].copy().reset_index(drop=True)

# =====================================================
#포트폴리오의 Value, DD 계산
# =====================================================
# 시간 데이터 길이
numData = Time.shape[0]

# Weight
W1 = Signal # risky asset 투자비중
W2 = 1 - W1 # risk free asset 의 투자비중

# portfolio value
Rp = pd.Series(np.zeros(numData)) # 수익률
Vp = pd.Series(np.zeros(numData))
Vp[0] = 100 # 초기 투자금 설정
for t in range(1,numData):
    Rp[t] = W1[t-1]*R1[t] + W2[t-1]*R2[t]
    Vp[t] = Vp[t-1]*(1 + Rp[t]/100)

# portfolio DD
MAXp = Vp.cummax()
DDp = (Vp/MAXp -1) * 100 # 단위 : %

# ====================================================
# 벤치마크의 Value, DD 계산
# ====================================================
Risky = rawRisky[idx:].copy().reset_index(drop=True)

# BM Value (투자 시작일 기준으로 표현됨)
Vb = (rawRisky[idx:]/rawRisky[idx]).copy().reset_index(drop=True)*100

# BM DD
MAXb= Vb.cummax()
DDb = (Vb/MAXb -1)*100 # 단위: %

# ====================================================
# 그래프 그리기
# ====================================================

# Value 만 그리기
plt.figure(figsize= (10, 5))
plt.plot(Time, Vp, label='Mom_12m')
plt.plot(Time, Vb, label='K200')
plt.legend()
plt.title('<Time-series Momentum Strategy>')
plt.xlabel('time')
plt.xlabel('value')
plt.show()

# Value 와 MDD 그리기 
fig  = plt.figure(figsize=(10,7))           #가로 세로 길이 지정
gs = gridspec.GridSpec(nrows=2,               #row의 개수
                       ncols=1,               # col 개수
                       height_ratios=[8,3],
                       width_ratios=[5])      # subplot의 크기를 서로 달리 지정

ax0 = plt.subplot(gs[0])
ax0.plot(Time, Vp, label='Mom_12')
ax0.plot(Time, Vb, label='Ks200')
ax0.set_title('<Value>')
ax0.grid(True)
ax0.legend()

ax1 = plt.subplot(gs[1])
ax1.plot(Time, DDp, label='Mom_12')
ax1.plot(Time, DDb, label='Ks200')
ax1.set_title('<Draw-down>')
ax1.grid(True)
ax1.legend()

plt.show()








