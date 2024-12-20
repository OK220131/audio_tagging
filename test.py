import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
import panns_inference
from panns_inference import AudioTagging, SoundEventDetection, labels
import sys
import tagging

def print_audio_tagging_result(clipwise_output):
    """イベント検出結果の出力(数値)

    Args:
      clipwise_output: (classes_num,)(検出情報)
    """
    sorted_indexes = np.argsort(clipwise_output)[::-1]

    # Print audio tagging top probabilities
    for k in range(10):
        print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]], 
            clipwise_output[sorted_indexes[k]]))
        
def plot_sound_event(framewise_output,sound_events=[16,17,22,23]):
    
    """イベントの検出結果をグラフしてplotする関数
    関数下部のコメントアウトを外すと円グラフがplotされる
    
    Args:
      framewise_output: (time_steps, classes_num)出力情報
      sound_events:検出するイベントのID。泣き声と笑い声が対象。違うものを検出したい場合は変更可能
    """

    #検出結果の保存先を弄る際は↓を変更(デフォルトは'results/{音声ファイル名}_result.png')
    args = sys.argv
    out_fig_path = 'results/'+ args[1]+'_result.png'
    os.makedirs(os.path.dirname(out_fig_path), exist_ok=True)

    classwise_output = np.max(framewise_output, axis=0) # (classes_num,)
    #print("classwise_output:",classwise_output)
    idxes = np.argsort(classwise_output)[::-1]
    idxes = sound_events
    print("idxes:",idxes)
    ix_to_lb = {i : label for i, label in enumerate(labels)}
    lines = []
    print("len:",len(framewise_output[:, idxes[0]]))
    for j in range(2):
        #print(idx,":",framewise_output[:, idx],"len:",len(framewise_output[:, idx]))
        avg=[]
        idx1=idxes[j*2]
        idx2=idxes[j*2+1]
        #for i in range(1,(len(framewise_output[:, idx])//64)+1):
            #avg.append((framewise_output[:, idx][i*31]+framewise_output[:, idx][i*63])/2)
        for i in range(1,(len(framewise_output[:, idx1])//32)+1):
            avg.append((framewise_output[:, idx1][i*31])+framewise_output[:, idx2][i*31])
        #print("avg:",avg)
        if j==0:
            avg1=avg
        else:
            avg2=avg
        line, = plt.plot(avg, label=ix_to_lb[idx1])
        lines.append(line)
    count=0
    count2=0
    for i in range(len(avg1)):
        if avg1[i]-avg2[i]>0.05 or avg2[i]-avg1[i]>0.05:
            count2+=1
            if avg1[i]<=avg2[i]:
                count+=1
    print("cry=",count/count2*100,"%,laugh=",(1-count/count2)*100,"%")
    plt.legend(handles=lines)
    plt.xlabel('Seconds')
    plt.ylabel('Probability')
    plt.ylim(0, 1.)

    #↓は円グラフのプロットで使用
    #x = np.array([count/count2*100,(1-count/count2)*100])
    #plt.pie(x,labels=["cry","laugh"],autopct='%.1f%%', startangle=90)
    plt.savefig(out_fig_path)
    print('Save fig to {}'.format(out_fig_path))
    
    
def audio_tagging(audio_path,sound_events=[16,17,22,23]):
    """音声認識を行う関数(これを実行するとグラフ出力まで)
    
    引数:
    audio_path:オーディオが存在するパス
    sound_events:検出するイベントのID。泣き声と笑い声が対象。違うものを検出したい場合は変更可能
    """
    device = 'cuda' # 'cuda' | 'cpu'
    print("load Model")
    (audio, _) = librosa.core.load(audio_path, sr=32000, mono=True)#モデルのロード
    audio = audio[None, :]  # (batch_size, segment_samples)

    print('------ Audio tagging ------')
    at = tagging.AudioTagging(checkpoint_path=None, device=device)
    (clipwise_output, embedding) = at.inference(audio)
    """clipwise_output: (batch_size, classes_num), embedding: (batch_size, embedding_size)"""
    #print("clipwise:",clipwise_output[0][73])
    print_audio_tagging_result(clipwise_output[0])

    print('------ Sound event detection ------')
    sed = tagging.SoundEventDetection(
        checkpoint_path=None, 
        device=device, 
        interpolate_mode='nearest', # 'nearest'
    )
    framewise_output = sed.inference(audio)
    """(batch_size, time_steps, classes_num)"""

    #plot_sound_event_detection_result(framewise_output[0])
    plot_sound_event(framewise_output[0],sound_events)


if __name__ == '__main__':
    #引数に分析ファイル名を指定
    args = sys.argv
    print(args)
    #audio_tagging(audio_path = '泣き声.mp3')
    #audio_tagging(audio_path = '笑い声.mp3')
    audio_tagging(audio_path = args[1])