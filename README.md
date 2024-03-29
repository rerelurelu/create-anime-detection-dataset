# キャラクター識別モデルの作成について

# 内容

[私のポートフォリオ](https://zwanywhere.com/)で使用されているキャラクター識別モデルの作成に使用したファイルです。個々のファイルについて詳しい説明はありませんが、データセットや識別モデルの作成とweb上での処理の大まかな流れを説明したいと思います。

# データセット作成の流れ

<ol style="list-style-type: decimal">
    <li>動画ファイルからキャラクターの顔画像だけを保存</li>
    <li>得られた顔画像をキャラクター別に分ける(手作業)</li>
    <li>正解ラベルと画像をNumpyファイルに変換(データセット作成完了)</li>
    <li>モデルの構造を定義して学習</li>
    <li>出来上がったモデルに満足したらモデルを保存して完成</li>
</ol>
<br>
<br>
※補足

1 について、アニメキャラクターの顔検出自体は他の方によって作成された OpenCV で使用できるカスケード分類器があるので、それを使用してアニメに登場する全てのキャラクターの顔画像を検出して保存します。

# web 上での処理

サイト上では入力された画像から補足で紹介したカスケード分類器によって画像に描かれているアニメキャラクターの顔画像を検出し、検出された顔画像一枚一枚に関して自作のモデルを使ってキャラクターの名前を推測します。推測結果は名前と確率が返される様になっており（例えば、A=70%,B=20%,C=10%なら「A」と「70%」が返される）、推測の確率が閾値を超えていれば返された結果を画像に描写し、閾値以下ならばデータセットにいないキャラクターとして画像に描写します。そして、結果が描写された画像を画面上に表示します。
