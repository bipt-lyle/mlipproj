import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf

def load_and_preprocess_data():
    # URL列表
    import requests
    import os
    import zipfile
    import pandas as pd
    import numpy as np
    import glob
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    from sklearn.preprocessing import StandardScaler
    urls = [
        "https://media.fdj.fr/static-draws/csv/euromillions/euromillions_202002.zip",
        "https://media.fdj.fr/static-draws/csv/euromillions/euromillions_201902.zip",
        "https://media.fdj.fr/static-draws/csv/euromillions/euromillions_201609.zip",
        "https://media.fdj.fr/static-draws/csv/euromillions/euromillions_201402.zip",
        "https://media.fdj.fr/static-draws/csv/euromillions/euromillions_201105.zip",
        "https://media.fdj.fr/static-draws/csv/euromillions/euromillions_200402.zip"
    ]
    # 存储路径
    save_path = 'data'  # 请根据实际情况调整路径
    # 检查目标目录是否存在，如果不存在，则创建
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 下载并解压文件
    for url in urls:
        file_name = url.split('/')[-1]
        full_path = os.path.join(save_path, file_name)

        # 发起请求下载文件
        response = requests.get(url)
        if response.status_code == 200:
            # 写入文件到指定路径
            with open(full_path, 'wb') as file:
                file.write(response.content)
            print(f"文件已下载并保存到：{full_path}")
            
            # 解压文件
            with zipfile.ZipFile(full_path, 'r') as zip_ref:
                zip_ref.extractall(save_path)
            print(f"文件已解压到：{save_path}")
            
            # 删除原zip文件
            os.remove(full_path)
            print(f"原zip文件已删除：{full_path}")
        else:
            print(f"下载失败，状态码：{response.status_code}")
    folder_path = 'data'
    csv_files = glob.glob(f'{folder_path}/*.csv')
    columns_to_keep = ['annee_numero_de_tirage', 'boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'etoile_1', 'etoile_2']
    df = pd.DataFrame()
    for file in csv_files:
        # Read the CSV file, only keeping specified columns and setting encoding, separator, and index column parameters
        df_temp = pd.read_csv(file, encoding='ISO-8859-1', sep=';', index_col=False, usecols=columns_to_keep)
        # Append the data to the combined DataFrame
        df = pd.concat([df, df_temp], ignore_index=True)

    # Sort the combined DataFrame by 'annee_numero_de_tirage' in ascending order
    df.sort_values(by='annee_numero_de_tirage', inplace=True)
    # Reset the index of the combined DataFrame
    df.reset_index(drop=True, inplace=True)
    df.iloc[:, 1:6] = np.sort(df.iloc[:, 1:6].values, axis=1)
    # 对星号号码进行排序
    df.iloc[:, 6:8] = np.sort(df.iloc[:, 6:8].values, axis=1)
    df['year'] = df['annee_numero_de_tirage'].apply(lambda x: str(x)[:4])
    return df


def EDA_plots(df):
    from io import BytesIO
    import base64
    import plotly
    from plotly.offline import plot
    plots_divs = []

    # 使用matplotlib和seaborn绘制箱线图，并保存为HTML字符串
    plt.figure(figsize=(14, 7))
    sns.boxplot(data=df[['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'etoile_1', 'etoile_2']])
    plt.title('Boxplot of Lottery Numbers')
    tmpfile = BytesIO()
    plt.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    html_img = f'<img src=\'data:image/png;base64,{encoded}\'>'
    plots_divs.append(html_img)
    plt.close()

    # 使用Plotly绘制主球号和星号的频率分布图，并将图表转换为HTML div
    for i in range(1, 6):  # 主球号
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df[f'boule_{i}'], nbinsx=50, marker_color='blue', name=f'boule_{i}'))
        fig.update_layout(
            title_text=f'Frequency of boule_{i}',
            xaxis_title_text='Number',
            yaxis_title_text='Frequency',
            bargap=0.2,
        )
        plot_div = plot(fig, output_type='div', include_plotlyjs=False)
        plots_divs.append(plot_div)

    for i in range(1, 3):  # 星号
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df[f'etoile_{i}'], nbinsx=12, marker_color='red', name=f'etoile_{i}'))
        fig.update_layout(
            title_text=f'Frequency of etoile_{i}',
            xaxis_title_text='Number',
            yaxis_title_text='Frequency',
            bargap=0.2,
        )
        plot_div = plot(fig, output_type='div', include_plotlyjs=False)
        plots_divs.append(plot_div)
    
    main_balls = df[['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']].values.flatten()
    star_balls = df[['etoile_1', 'etoile_2']].values.flatten()
    # 绘制主球号码的频率分布
    fig_main = go.Figure()
    fig_main.add_trace(go.Histogram(x=main_balls, nbinsx=50, marker_color='blue', name='Main Balls'))
    fig_main.update_layout(
        title_text='Frequency Distribution of Main Ball Numbers',
        xaxis_title_text='Number',
        yaxis_title_text='Frequency',
        bargap=0.2,  # 间距
    )
    plot_div_main = plot(fig_main, output_type='div', include_plotlyjs=False)
    plots_divs.append(plot_div_main)

    # 绘制星号球号码的频率分布
    fig_star = go.Figure()
    fig_star.add_trace(go.Histogram(x=star_balls, nbinsx=12, marker_color='red', name='Star Balls'))
    fig_star.update_layout(
        title_text='Frequency Distribution of Star Ball Numbers',
        xaxis_title_text='Number',
        yaxis_title_text='Frequency',
        bargap=0.2,  # 间距
    )
    plot_div_star = plot(fig_star, output_type='div', include_plotlyjs=False)
    plots_divs.append(plot_div_star)

    corr_matrix = df[['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']].corr()

    # 绘制热力图并保存为图片
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt=".2f")
    plt.title('Heatmap of Correlation Between Main Balls')
    tmpfile = BytesIO()
    plt.savefig(tmpfile, format='png')
    plt.close()  
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    html_img = f'<img src=\'data:image/png;base64,{encoded}\'>'

    plots_divs.append(html_img)

    num_range = 50
    co_occurrence_matrix = np.zeros((num_range, num_range))

    for index, row in df.iterrows():
        for i in range(1, 6):
            for j in range(i+1, 6):
                num1, num2 = sorted([row[f'boule_{i}'], row[f'boule_{j}']])
                co_occurrence_matrix[num1-1, num2-1] += 1

    for i in range(num_range):
        for j in range(i+1, num_range):
            co_occurrence_matrix[j, i] = co_occurrence_matrix[i, j]

    hover_text = [['number1: {}, number2: {}, frequency: {}'.format(i+1, j+1, co_occurrence_matrix[i, j]) for j in range(num_range)] for i in range(num_range)]

    fig = go.Figure(data=go.Heatmap(
        z=co_occurrence_matrix,
        x=[str(i) for i in range(1, num_range + 1)],
        y=[str(i) for i in range(1, num_range + 1)],
        hoverongaps=False,
        colorscale='Viridis',
        colorbar=dict(title='Co-occurrence Frequency'),
        text=hover_text,
        hoverinfo='text'
    ))

    fig.update_layout(
        title='Heatmap of Number Pair Co-occurrences',
        xaxis=dict(title='Number', tickmode='array', tickvals=list(range(1, num_range + 1))),
        yaxis=dict(title='Number', tickmode='array', tickvals=list(range(1, num_range + 1))),
        width=1000,
        height=800,
    )

    plot_div = plot(fig, output_type='div', include_plotlyjs=False)
    plots_divs.append(plot_div)

    # 绘制每年平均号码的变化趋势
    df['year'] = df['annee_numero_de_tirage'].apply(lambda x: str(x)[:4])
    average_numbers_per_year = df.groupby('year').mean()
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=average_numbers_per_year[['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'etoile_1', 'etoile_2']])
    plt.title('Average Lottery Numbers per Year')
    plt.ylabel('Average Number')
    plt.xlabel('Year')
    tmpfile = BytesIO()
    plt.savefig(tmpfile, format='png')
    plt.close() 
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    html_img = f'<img src=\'data:image/png;base64,{encoded}\'>'
    plots_divs.append(html_img)

    # 初始化存储每年最多出现号码的 DataFrame
    most_frequent_numbers_per_year = pd.DataFrame()
    # 对每个球号进行操作
    for ball in ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'etoile_1', 'etoile_2']:
        most_frequent_numbers = []
        for year in df['year'].unique():
            year_df = df[df['year'] == year]
            most_frequent_number = year_df[ball].value_counts().idxmax()
            most_frequent_numbers.append(most_frequent_number)
        
        most_frequent_numbers_per_year[ball] = most_frequent_numbers
    most_frequent_numbers_per_year.index = df['year'].unique()
    # 绘制每年最频繁出现的数字的变化趋势
    plt.figure(figsize=(14, 7))
    for ball in most_frequent_numbers_per_year.columns:
        sns.lineplot(data=most_frequent_numbers_per_year[ball], label=ball)
    plt.title('Most Frequent Lottery Numbers per Year')
    plt.ylabel('Most Frequent Number')
    plt.xlabel('Year')
    plt.legend(title='Ball Number')
    plt.xticks(rotation=45)
    tmpfile = BytesIO()
    plt.savefig(tmpfile, format='png')
    plt.close()  # 关闭 plt，避免内存泄露
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    html_img = f'<img src=\'data:image/png;base64,{encoded}\'>'
    plots_divs.append(html_img)
    save_plots_divs_to_file(plots_divs, 'EDA_plots.html')
    return plots_divs

def standard(df):
    from sklearn.preprocessing import StandardScaler
    import joblib
    
    data_dir = 'data_for_model'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
   
    df = df.drop(['annee_numero_de_tirage','year'], axis=1)
    scaler = StandardScaler().fit(df.values)
    joblib.dump(scaler, 'data_for_model/scaler.save')
    transformed_dataset = scaler.transform(df.values)
    transformed_df = pd.DataFrame(data=transformed_dataset, index=df.index)
    return df,transformed_dataset,transformed_df, scaler



def evaluate_predict_plots():
    from io import BytesIO
    import base64
    import plotly
    from plotly.offline import plot
    import pickle
    import joblib
    file_path = 'data_for_model/data.csv'
    df = pd.read_csv(file_path)
    scaler = joblib.load('data_for_model/scaler.save')
    file2_path = 'data_for_model/transformed_data.csv'
    transformed_df = pd.read_csv(file2_path)
    
    train,label,test,test_label,val,val_label,number_of_features = train_test_data(df,transformed_df)

    model = load_model('model/euromillions.h5')
    history_path = 'model/euromillions_history.pkl'
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    plots_divs = []
#模型评估图：
    plt.figure()
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    
    # 将图片保存到临时内存中，而不是文件系统
    tmpfile = BytesIO()
    plt.savefig(tmpfile, format='png')
    plt.close()  # 关闭 plt，避免内存泄露
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    html_img = f'<img src=\'data:image/png;base64,{encoded}\'>'

    # 将图表的 Base64 编码的图片添加到图表列表中
    plots_divs.append(html_img)
#模型评估：
    mse, rmse, mae, r2,test_label_original,predicted_output = evaluate_model(model, test, test_label, scaler)
    metrics = {
    'MSE': mse,
    'RMSE': rmse,
    'MAE': mae,
    'R2': r2
    }

    # 可视化预测结果
    plt.figure(figsize=(15, 30))
    for i in range(number_of_features):
        plt.subplot(number_of_features, 1, i+1)  # 动态决定子图布局
        plt.plot(test_label_original[:, i], label='Actual')
        plt.plot(predicted_output[:, i], label='Predicted')
        plt.title(f'Feature {i+1}')
        plt.legend()
    tmpfile = BytesIO()
    plt.savefig(tmpfile, format='png')
    plt.close()  # 确保关闭图表以释放内存
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    html_img = f'<img src=\'data:image/png;base64,{encoded}\' style="width:100%;">'
    plots_divs.append(html_img)


    # 计算每个球的预测准确率
    ball_names = ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'etoile_1', 'etoile_2']
    accuracies = []
    number_of_features = len(ball_names)
    for i in range(number_of_features):
        correct_predictions = np.sum(test_label_original[:, i] == predicted_output[:, i])
        total_predictions = len(test_label_original[:, i])
        accuracy = correct_predictions / total_predictions * 100
        accuracies.append(accuracy)

    # 绘制预测准确率的可视化
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab20.colors
    for i, (accuracy, color) in enumerate(zip(accuracies, colors)):
        plt.barh(i, accuracy, color=color, label=ball_names[i])
        plt.text(accuracy, i, f'{accuracy:.1f}%', va='center')
    plt.xlim(0, 100)
    plt.yticks(range(number_of_features), ball_names)
    plt.xlabel('Accuracy (%)')
    plt.title('Prediction Accuracy for Each Ball')
    plt.grid(axis='x')
    plt.legend()
    plt.gca().invert_yaxis()
    tmpfile = BytesIO()
    plt.savefig(tmpfile, format='png')
    plt.close()  # 确保关闭图表以释放内存
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    html_img = f'<img src=\'data:image/png;base64,{encoded}\' style="width:100%;">'
    plots_divs.append(html_img)
    #预测结果：
    predictions = make_predictions(df,scaler,model)
    predictions_html = predictions.to_html(classes='predictions',index=False)
    save_plots_divs_to_file(plots_divs, 'evaluate_predict.html')
    return plots_divs, metrics,predictions_html


def train_test_data(df,transformed_df):
    number_of_rows = df.values.shape[0]
    window_length = 5
    number_of_features = df.values.shape[1]
    train_size = int(number_of_rows * 0.8)
    test_size = int((number_of_rows - train_size) * 0.5)  # 50% 划分给测试集，50% 划分给验证集

    train_data, test_val_data = transformed_df.iloc[:train_size], transformed_df.iloc[train_size:]
    test_data, val_data = train_test_split(test_val_data, test_size=test_size, shuffle=False)

    # 构建训练集和标签集
    train = np.empty([train_size - window_length, window_length, number_of_features], dtype=float)
    label = np.empty([train_size - window_length, number_of_features], dtype=float)

    for i in range(0, train_size - window_length):
        train[i] = train_data.iloc[i:i+window_length, 0:number_of_features]
        label[i] = train_data.iloc[i+window_length:i+window_length+1, 0:number_of_features]

    # 构建测试集和标签集
    test = np.empty([test_size - window_length, window_length, number_of_features], dtype=float)
    test_label = np.empty([test_size - window_length, number_of_features], dtype=float)

    for i in range(0, test_size - window_length):
        test[i] = test_data.iloc[i:i+window_length, 0:number_of_features]
        test_label[i] = test_data.iloc[i+window_length:i+window_length+1, 0:number_of_features]

    # 构建验证集和标签集
    val = np.empty([len(val_data) - window_length, window_length, number_of_features], dtype=float)
    val_label = np.empty([len(val_data) - window_length, number_of_features], dtype=float)

    for i in range(0, len(val_data) - window_length):
        val[i] = val_data.iloc[i:i+window_length, 0:number_of_features]
        val_label[i] = val_data.iloc[i+window_length:i+window_length+1, 0:number_of_features]
    return train,label,test,test_label,val,val_label,number_of_features


def build_and_train_model(number_of_features,train,label,val,val_label):
    import os
    import pickle
    window_length = 5
    if os.path.exists('model/euromillions.h5'):
        model = load_model('model/euromillions.h5')
        history_path = 'model/euromillions_history.pkl'
        with open(history_path, 'rb') as f:
            history = pickle.load(f)

    else:
        model = Sequential()
        model.add(LSTM(64, input_shape=(window_length, number_of_features), return_sequences=True))
        model.add(Dropout(0.2))
        # model.add(LSTM(64, return_sequences=True))
        # model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(number_of_features))
        #模型编译和训练
        model.compile(loss='mse', optimizer='rmsprop')
        history = model.fit(train, label, validation_data=(val, val_label), batch_size=64, epochs=120)
        # 保存模型
        model.save('model/euromillions.h5')
        history_path = 'model/euromillions_history.pkl'
        with open(history_path, 'wb') as f:
            pickle.dump(history.history, f)
        return model,history
    return model, history

def evaluate_model(model, test, test_label, scaler):
    # 模型预测
    scaled_predicted_output = model.predict(test)
    
    # 将预测结果转换回原始比例
    predicted_output = scaler.inverse_transform(scaled_predicted_output).astype(int)
    
    # 将测试标签转换回原始比例
    test_label_original = scaler.inverse_transform(test_label)
    
    # 计算MSE
    mse = mean_squared_error(test_label_original, predicted_output)
    
    # 计算RMSE
    rmse = np.sqrt(mse)
    
    # 计算MAE
    mae = mean_absolute_error(test_label_original, predicted_output)
    
    # 计算R²
    r2 = r2_score(test_label_original, predicted_output)
    
    return mse, rmse, mae, r2,test_label_original,predicted_output




def make_predictions(df,scaler,model):
    # 预测部分
    to_predict = df.iloc[-5:]
    scaled_to_predict = scaler.transform(to_predict)

    scaled_predicted_output_1 = model.predict(np.array([scaled_to_predict]))
    data = scaler.inverse_transform(scaled_predicted_output_1).astype(int)
    predict = pd.DataFrame(data, columns=['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'etoile_1', 'etoile_2'])
    return predict

def save_plots_divs_to_file(plots_divs, filename):
    import os
    # 确保 statics 文件夹存在
    statics_path = 'statics'
    if not os.path.exists(statics_path):
        os.makedirs(statics_path)
    
    # 完整的文件路径
    file_path = os.path.join(statics_path, filename)
    
    # 写入文件，并覆盖之前的内容
    with open(file_path, 'w') as file:
        for div in plots_divs:
            file.write(div + '\n')
def main():

    # Load and preprocess the data
    print('begin load')
    df= load_and_preprocess_data()
    print("begin data")
    df,transformed_dataset,transformed_df, scaler = standard(df)

    train,label,test,test_label,val,val_label,number_of_features = train_test_data(df,transformed_df)
    print("build model")

    model,history = build_and_train_model(number_of_features,train,label,val,val_label)

    print("evaluate")

    # Evaluate the model
    mse, rmse, mae, r2,test_label_original,predicted_output = evaluate_model(model, test, test_label, scaler)
    print(f"均方误差（MSE）: {mse}")
    print(f"均方根误差（RMSE）: {rmse}")
    print(f"平均绝对误差（MAE）: {mae}")
    print(f"决定系数（R²）: {r2}")

    # Make predictions (assuming you have to_predict ready based on your code)
    predictions = make_predictions(df,scaler,model)

    # Print or use the predictions
    print(predictions)



if __name__ == "__main__":
    main()
