from models.crf import CRFModel
from utils import load_model, extend_maps, prepocess_data_for_lstmcrf, save_model
from data import build_corpus
from evaluating import Metrics
from evaluate import ensemble_evaluate, crf_train_eval
import csv

HMM_MODEL_PATH = './ckpts/hmm.pkl'
CRF_MODEL_PATH = './ckpts/crf.pkl'
BiLSTM_MODEL_PATH = 'ckpts/old_bilstm.pkl'
BiLSTMCRF_MODEL_PATH = 'ckpts/old_bilstm_crf.pkl'
CRF_Record_PATH='./train/CRF_record.csv'

REMOVE_O = False  # 在评估的时候是否去除O标记


def main():
    # f=open(CRF_Record_PATH,'w')
    f = open(CRF_Record_PATH, 'a')
    csv_writer = csv.writer(f)
    # csv_writer.writerow(["algorithm", "c1", "c2","max_iterations","precision","recall","f1_score"])

    print("读取数据...")
    train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus("train")
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

    # print("加载并评估hmm模型...")
    # hmm_model = load_model(HMM_MODEL_PATH)
    # hmm_pred = hmm_model.test(test_word_lists,
    #                           word2id,
    #                           tag2id)
    # metrics = Metrics(test_tag_lists, hmm_pred, remove_O=REMOVE_O)
    # metrics.report_scores()  # 打印每个标记的精确度、召回率、f1分数
    # metrics.report_confusion_matrix()  # 打印混淆矩阵

    # 训练crf模型
    crf_model = CRFModel()
    c1=0.7
    c2=0.7
    itegration=100
    gap=0.05
    alg='lbfgs'

    # for i in range(20):
    #     c1+=gap
    #     c1
    #     crf_model.set(alg,c1,c2,itegration)
    #     print(crf_model.model.c1,crf_model.model.c2,crf_model.model.max_iterations)
    #     crf_model.train(train_word_lists, train_tag_lists)
    #     # save_model(crf_model, CRF_MODEL_PATH)
    #     # 加载并评估CRF模型
    #     print("加载并评估crf模型...")
    #     # crf_model = load_model(CRF_MODEL_PATH)
    #     crf_pred = crf_model.test(test_word_lists)
    #     metrics = Metrics(test_tag_lists, crf_pred, remove_O=REMOVE_O)
    #     metrics.report_scores()
    #     # metrics.report_confusion_matrix()
    #     avg_metrics=metrics._cal_weighted_average()
    #     csv_writer.writerow([alg,c1,c2,itegration,
    #                          avg_metrics['precision'],
    #                          avg_metrics['recall'],
    #                          avg_metrics['f1_score']])
    # crf_model.set(alg, c1, c2, itegration)
    # crf_pred = crf_train_eval(
    #     (train_word_lists, train_tag_lists),
    #     (test_word_lists, test_tag_lists)
    # )



if __name__ == "__main__":
    main()
