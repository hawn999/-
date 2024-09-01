from models.crf import CRFModel
from utils import load_model, extend_maps, prepocess_data_for_lstmcrf, save_model
from data import build_corpus
from evaluating import Metrics
from evaluate import ensemble_evaluate, bilstm_train_and_eval
import csv
import matplotlib.pyplot as plt

HMM_MODEL_PATH = './ckpts/hmm.pkl'
CRF_MODEL_PATH = './ckpts/crf.pkl'
BiLSTM_MODEL_PATH = 'ckpts/old_bilstm.pkl'
BiLSTMCRF_MODEL_PATH = 'ckpts/old_bilstm_crf.pkl'
CRF_Record_PATH='./train/BiLSTM_record.csv'

REMOVE_O = False  # 在评估的时候是否去除O标记
x1=[]#batch_size
x2=[0.001]#lr
y1=[]
y2=[]
y3=[]

def main():
    # # f=open(CRF_Record_PATH,'w')
    # f = open(CRF_Record_PATH, 'a')
    # csv_writer = csv.writer(f)
    # # csv_writer.writerow(["algorithm", "c1", "c2","max_iterations","precision","recall","f1_score"])

    bilstm_model.set_batch_size(32)
    func()
    bilstm_model.set_batch_size(64)
    func()
    bilstm_model.set_batch_size(128)
    func()
    bilstm_model.set_batch_size(256)
    func()

    plt.plot(x1, y1, 'red')
    plt.plot(x1, y2, 'green')
    plt.plot(x1, y3, 'blue')
    # print("start drawing")
    plt.xlabel("batch_size")
    plt.ylabel("value")
    plt.title("lr=0.001  red:precision  green:recall  blue:F1_score")
    plt.show()

def func():
    # bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
    bilstm_model.model.bilstm.flatten_parameters()  # remove warning
    lstm_pred = bilstm_train_and_eval(
        (train_word_lists, train_tag_lists),
        (dev_word_lists, dev_tag_lists),
        (test_word_lists, test_tag_lists),
        bilstm_word2id, bilstm_tag2id,
        crf=False
    )
    lstm_pred, target_tag_list = bilstm_model.test(test_word_lists, test_tag_lists,
                                           bilstm_word2id, bilstm_tag2id)
    metrics = Metrics(target_tag_list, lstm_pred, remove_O=REMOVE_O)
    avg_metrics = metrics._cal_weighted_average()

    x1.append(bilstm_model.batch_size)
    y1.append(avg_metrics['precision'])
    y2.append(avg_metrics['recall'])
    y3.append(avg_metrics['f1_score'])
    metrics.report_scores()
    metrics.report_confusion_matrix()



if __name__ == "__main__":
    print("读取数据...")
    train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus("train")
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

    bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
    bilstm_model = load_model(BiLSTM_MODEL_PATH)
    main()
