# AntMaze trajectory comparison, adopted TD3+RAPO cell

medium-diverse 그림과 `amo_log` main (`td3_amo_bootrms_maincand`)에서 맞춘 셀은 α_E=α_B=5, α_lr=3e-4이다. π_E loss는 `-B_π` (`execution_score=bpi`, `execution_meta_loss` 기본값 `le`), π_B loss는 `L2_RMS` (`bootstrap_loss=l2_rms`)이다.

이전에 올린 세 환경 그림은 α=1, α_lr=3e-4였다. 그 셀의 main 최종 점수는 0–5%라서 여기 숫자와 다르다. 이번 파일은 α=5, α_lr=3e-4, seed 0–3, step 1,000,000만 사용했다.

같은 α=5, α_lr=3e-4의 Shared actor `step_1000000.npz`는 medium-play, large-play, large-diverse에 없다. α=1 Shared actor로 짝을 만들지 않았다.

main 100-episode 최종 점수 (참고): medium-play 72.5 ± 14.9, large-play 29.5 ± 13.4, large-diverse 43.5 ± 5.3. 이 궤적은 seed당 25 episode다.
