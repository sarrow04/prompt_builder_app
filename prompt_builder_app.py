# --------------------------------------------------------------------------
# プロンプト生成ロジック部（✨✨ この部分を改善 ✨✨）
# --------------------------------------------------------------------------
def generate_prompt(
    problem_type, source_type, analysis_goal, data_context,
    train_path, test_path, submit_path, single_path,
    target_col, id_col, time_col,
    models, use_ensemble, tune_hyperparams,
    ts_features, include_corr, include_scaling, graphs,
    save_dir_path
    ):
    """ユーザーの選択に基づいてAIへのプロンプトを生成する関数"""
    # ... （関数の前半部分は変更なし） ...
    prompt = [
        f"### 依頼内容：機械学習を用いた「{problem_type}」の「{analysis_goal}」",
        "これからGoogle Colab環境で実行する、データ分析のPythonコードをステップ・バイ・ステップで生成してください。",
        "### 指示の前提条件:",
        "- **再現性の確保**: 分析の再現性が取れるよう、モデルの `random_state` は `42` に固定してください。",
        "- **可読性の向上**: 適切なコメントを追加し、可能であれば処理を関数にまとめてコードの可読性を高めてください。",
        "\n# ==================================",
        "# Step 1: 環境設定とデータ読み込み",
        "# ==================================",
        "### ライブラリのインストール",
        "!pip install japanize-matplotlib lightgbm shap holidays scikit-learn -q",
        "\n### 主要ライブラリのインポート",
        "# (pandas, numpy, lightgbm, sklearn, matplotlib, seaborn, shap, holidays, os などをインポートしてください)",
        "\n### Google Driveのマウント",
        "from google.colab import drive",
        "drive.mount('/content/drive')",
        "\n### ファイルパスと保存先の設定",
    ]
    
    # ... (ファイルパス設定の部分は変更なし) ...
    if source_type == "Kaggle形式":
        prompt.append(f"train_data_path = '{train_path}'")
        prompt.append(f"test_data_path = '{test_path}'")
        prompt.append(f"submit_template_path = '{submit_path}'")
    else:
        prompt.append(f"file_path = '{single_path}'")
    prompt.append(f"save_folder_path = '{save_dir_path}'")
    
    # ... (データ読み込み、概要把握の部分は変更なし) ...
    prompt.append("\n### データの読み込み")
    if source_type == "Kaggle形式":
        prompt.append("df_train = pd.read_csv(train_data_path)")
        prompt.append("df_test = pd.read_csv(test_data_path)")
    else:
        prompt.append("df = pd.read_csv(file_path)")

    prompt.extend([
        "\n# ==================================",
        "# Step 2: データ概要の把握",
        "# ==================================",
        "### データのカラム名とサンプル",
        "以下に、分析対象となるデータのカラム構成と先頭数行のサンプルを示します。これを元に分析を進めてください。",
        "--- データ概要 ---",
        data_context,
        "--- ここまで ---",
        f"\n今回の分析の目的変数は `{target_col}` です。"
    ])

    prompt.extend([
        "\n# ==================================",
        f"# Step 3: {analysis_goal}のための特徴量エンジニアリングと前処理",
        "# ==================================",
        "### 実行してほしいタスク:",
    ])

    tasks = [f"- **保存先ディレクトリの作成**: `os.makedirs(save_folder_path, exist_ok=True)` を実行してください。"]
    
    ### --- 改善点: 前処理の指示をより具体的に --- ###
    # データ結合の指示
    if source_type == "Kaggle形式":
        tasks.append("- **データ結合**: `df_train`と`df_test`を一度結合し、共通の前処理を実装してください。処理後、再度`train`と`test`に分割する流れでお願いします。")

    # 時系列の前処理
    if problem_type == "時系列予測":
        tasks.append(f"- **時系列データの前処理**: `{time_col}`列をdatetime型に変換してください。")
        if ts_features:
            tasks.append(f"- **時系列特徴量の作成**: 以下の特徴量を作成してください。\n  - " + "\n  - ".join(ts_features))
        # ARIMA/Prophetが選択された場合の特別指示
        if "ARIMA" in models or "Prophet" in models:
             tasks.append("- **モデル特化の前処理**: ARIMAのために系列の定常性チェック（ADF検定など）と、必要であれば差分処理を実装してください。Prophetのためには、カラム名を`ds`と`y`に変更する処理も加えてください。")
    # 時系列以外の前処理
    else:
        tasks.append("- **カテゴリカル変数の処理**: **各カテゴリカル変数のユニーク数を調査**し、その数に応じてワンホットエンコーディングかラベルエンコーディングを適切に使い分けてください。")

    # 共通の前処理タスク
    tasks.append("- **欠損値処理**: 欠損値の有無を確認し、もし存在すれば適切な方法（例: 平均値、中央値、最頻値などで補完）で処理してください。")
    if include_scaling:
        tasks.append("- **特徴量スケーリング**: `StandardScaler`などを用いて、数値特徴量のスケールを揃える処理を実装してください。")
    if include_corr:
        tasks.append("- **相関分析**: 前処理後の特徴量間の相関行列を計算し、ヒートマップで可視化してください。")

    prompt.append("\n".join(tasks)) # ここで一度タスクリストをプロンプトに追加

    ### --- 改善点: モデル構築以降のステップを明確に分離 --- ###
    prompt.extend([
        "\n# ==================================",
        f"# Step 4: {analysis_goal}のためのモデル構築と評価",
        "# ==================================",
        "### 実行してほしいタスク:",
    ])
    
    tasks = [] # タスクリストをリセット

    # モデル学習
    if analysis_goal == "要因分析":
        tasks.append("- **モデル学習**: **解釈性の高い**モデル（ロジスティック回帰/線形回帰, 決定木など）を優先して使用してください。")
        tasks.append(f"- **要因分析**: モデルの係数やSHAP値を使い、「どの特徴量が `{target_col}` に正または負の影響を与えているか」を分析・考察してください。")
    else:
        tasks.append(f"- **モデル学習**: `{', '.join(models)}` を使って、予測精度が最大になるようにモデルを学習させてください。")
        if tune_hyperparams and models and not any(m in ["ARIMA", "Prophet"] for m in models):
            cv_method = "TimeSeriesSplitを使ったクロスバリデーション" if problem_type == "時系列予測" else "通常のクロスバリデーション(cv=5)"
            tasks.append(f"- **ハイパーパラメータチューニング**: GridSearchCVを使い、`{models[0]}`のモデルの予測精度をさらに向上させてください。({cv_method})")
        if use_ensemble and len(models) > 1:
            tasks.append("- **アンサンブル**: 学習させた複数のモデルの予測値を平均し、アンサンブル予測を行ってください。")

    # モデル評価
    evaluation_items = []
    if problem_type == "分類":
        evaluation_items = ["混同行列 (Confusion Matrix)", "正解率 (Accuracy), 適合率 (Precision), 再現率 (Recall), F1スコア", "AUCスコア"]
    elif problem_type == "回帰":
        evaluation_items = ["RMSE (Root Mean Squared Error)", "MAE (Mean Absolute Error)", "R2スコア (決定係数)"]
    elif problem_type == "時系列予測":
        evaluation_items = ["RMSE (Root Mean Squared Error)", "MAE (Mean Absolute Error)", "MAPE (Mean Absolute Percentage Error)"]
    if evaluation_items:
        tasks.append(f"\n- **モデル評価**: テストデータ（または検証データ）に対して、以下の指標を計算し、結果を報告してください。\n  - " + "\n  - ".join(evaluation_items))

    # 可視化
    if "特徴量の重要度 (SHAP)" not in graphs and not any(m in ["ARIMA", "Prophet"] for m in models):
        graphs.insert(0, "特徴量の重要度 (SHAP)")
    tasks.append(f"\n- **可視化**: 以下のグラフを生成し、`save_folder_path`に保存してください。\n  - " + "\n  - ".join(graphs))
    
    # 出力
    if source_type == "Kaggle形式":
        tasks.append(f"- **提出ファイルの作成**: `sample_submission.csv`の形式に合わせて、テストデータの予測結果を`os.path.join(save_folder_path, 'submission.csv')`として出力してください。ID列は`{id_col}`です。")
    elif problem_type == "時系列予測":
        tasks.append("- **未来予測**: 学習したモデルを使い、未来の予測値を算出し、実績値と合わせてグラフにプロットしてください。")

    prompt.append("\n".join(tasks))
    prompt.append("\n---\n以上の内容で、Pythonコードを生成してください。")
    return "\n".join(prompt)
