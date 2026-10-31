
# Dynamic Response Extension — Cances論文 1:1対応版 + Fe2MnAl

## 目的
`gkemlin/response-calculations-metals` の静的変分を動的 ε(ω) に拡張。

## ディレクトリ
- metals/Al40, metals/Fe2MnAl が元リポの構成と同じ (README参照)
- Fe2MnAl は Heusler: 擬ギャップ近傍に d共鳴 6極を持つ難例
- `src/response/adaptive_choice_extra_bands_dynamic.jl` は Section 6 / run_adaptive_algorithm.jl の動的版

## 静的->動的対応
- 静的 extra_bands ~ O(log(1/tol)*σ/gap)
- 動的 extra_bands(ω) ~ 共鳴 |ε_c-ε_v-ω| < η で追加が必要
  -> greedy_rational_shifts で ωgrid全体のmax errorがtol以下まで極追加

## 実行
julia --project=. metal/Al40/run_Al40.jl
julia --project=. metal/Fe2MnAl/run_Fe2MnAl.jl

## 比較相手
- 主敵: SoS = QE epsilon.x
- 補助: shifted Sternheimer (動的では条件数 (bandwidth+ω)/η で失速)
