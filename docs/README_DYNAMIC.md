
# Dynamic Response Extension — Cances論文 1:1対応版 + Fe2MnAl

## 目的
`gkemlin/response-calculations-metals` の静的変分を動的 ε(ω) に拡張。

## ディレクトリ
- metal/Al40, metal/Fe2MnAl が元リポの構成と同じ (README参照)
- Fe2MnAl は Heusler: 擬ギャップ近傍に d共鳴 6極を持つ難例
- `src/response/adaptive_choice_extra_bands_dynamic.jl` は Section 6 / run_adaptive_algorithm.jl の動的版
- 各 run スクリプトは a posteriori estimator で default 4 shifts と adaptive shifts を比較し、結果を `docs/*_adaptive_vs_default.csv` に保存

## 静的->動的対応
- 静的 extra_bands ~ O(log(1/tol)*σ/gap)
- 動的 N_extra(ω) は共鳴 `|ε_c-ε_v-ω| < η` で増加
  -> `greedy_rational_shifts_aposteriori` で ωgrid 全体の残差推定値が tol 以下になるまで極を追加

## 実行
julia --project=. metal/Al40/run_Al40.jl
julia --project=. metal/Fe2MnAl/run_Fe2MnAl.jl

両方のスクリプトで η = [0.2, 0.1, 0.05, 0.02] を評価する。保存 CSV の
`max_err` は exact solve による誤差ではなく、a posteriori residual estimate。

## 比較相手
- 主敵: SoS = QE epsilon.x
- 補助: shifted Sternheimer (動的では条件数 (bandwidth+ω)/η で失速)
