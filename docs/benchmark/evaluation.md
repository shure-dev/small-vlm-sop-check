# Evaluation policy

## Formal accuracy

正式なprecision、recall、F1、balanced accuracy、tIoUは、人手GT revisionを固定したevaluation runでのみ計算します。

prediction runとevaluation runを分けることで、後日GTが修正されても過去の予測を改変せず、どのGT版で測った数字かを追跡できます。

## Before human ground truth

人手GTがないunitでは、次だけを予備比較として扱えます。

- モデル間一致
- yes/no回答分布
- 境界差
- confidence分布

これらをaccuracy、正解率、教師一致とは表記しません。

## Split policy

分割はunit単位ではなくfactory/worker単位です。Factory Egoの現行unitは、選定・SOP設計・アノテーションの過程でモデルと管理者が閲覧するため、恒久的に `dev_seen` としtestへ昇格させません。真のtest splitは、誰も閲覧していないクリップに人手GTを付けて初めて作れます。
