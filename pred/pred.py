# -*- coding: GBK -*-
import json

import openpyxl
from keras_preprocessing.text import text_to_word_sequence

from data import build_corpus
from utils import load_model, extend_maps, prepocess_data_for_lstmcrf_test
import pandas as pd


# def pred():
train_word_lists=[]
train_tag_lists=[]
word2id={}
tag2id={}
bilstm_word2id={}
bilstm_tag2id={}
crf_word2id={}
crf_tag2id={}

# 导入word2id, tag2id
train_word_lists, train_tag_lists, word2id, tag2id = \
    build_corpus("train")
bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)

crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
bilstm_model = load_model('/Users/hwan/Downloads/named_entity_recognition-master/ckpts/bilstm_crf.pkl')
# bilstm_model = load_model('/Users/hwan/Downloads/named_entity_recognition-master/ckpts/bilstm.pkl')


# 读取Excel文件
# file_path = '/Users/hwan/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/830da9def43de92ae65513342e7367b1/Message/MessageTemp/137f4c83ecf5057aec6d9038c0c6a9f9/File/产学研数据1.xlsx'  # 替换为你文件的路径
file_path='/Users/hwan/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/830da9def43de92ae65513342e7367b1/Message/MessageTemp/137f4c83ecf5057aec6d9038c0c6a9f9/File/产学研数据2.xlsx'
df = pd.read_excel(file_path)

# 提取正文列的文本并存入列表
g_column_data = df['正文'].tolist()
results_list=[]
# 输出结果
# print(g_column_data)
for text in g_column_data:
    # 用txt构造test_word_list
    # text=''
    test_word_lists=[]
    # text = u'5月24日下午，清华大学科研院（先进技术院）一行在院长刘奕群教授的带领下来到北京大学参观调研，并与北京大学科学研究部、先进技术研究院、科技开发部、新工科建设办公室等部门开展学习贯彻习近平新时代中国特色社会主义思想主题教育联学共建。参观调研暨联学活动在昌平北大新校区举行，由北大科研部部长谢冰教授主持。北大科研部副部长兼党支部书记张存群、副部长李晓强，先进院副院长杨默函，科技开发部副部长郑如青，新工科办主任兼科研部副部长韦宇；清华大学科研院副院长李水清、蒋靖坤，先进技术院副院长高云峰等30余名党员同志参加活动。两校科研管理人员参观了北大新校区。在新校区管理委员会办公室刘红燕老师的带领下，他们先后参观了行政楼、公共教学楼、学生宿舍5号楼、教师公寓、计算机大楼，在参观交流中了解北大新校区的结构布局和功能定位。大家纷纷表示，北大新校区建筑气派、设备先进、功能齐备，体现出北京大学高质量发展的气魄和新工科建设蓬勃向上的势头。参观北大新校区参观之后，两校科研管理人员在计算机学院会议室以“开展高质量有组织科研，服务高水平科技自立自强”为主题进行交流座谈。座谈会现场座谈会上，张存群带领与会人员集体学习了习近平总书记关于高校科技创新工作的重要论述，重温习近平总书记考察北大、清华时的重要讲话精神。同志们表示，习近平总书记对加强高校科技创新工作的一系列重要论述高屋建瓴、举旗定向，对高水平研究型大学明确新的定位、提出新的要求。大家将牢记总书记在考察北京大学和清华大学时的殷殷嘱托，坚持办学正确政治方向，建设高素质教师队伍，形成高水平人才培养体系，提升原始创新能力，加快推进有组织科研，持续强化国家战略科技力量建设，以高水平科技创新服务高质量发展，为加快建设世界科技强国和实现中国式现代化作出新的贡献。韦宇、李水清、李晓强分别介绍了相关科研工作情况。座谈会发言（从左到右、从上到下依次为：张存群、韦宇、李水清、李晓强）韦宇介绍了北大新工科发展和新空间建设，以及北大-昌平产教融合创新中心的建设情况。北大新校区是北大发展新工科的主阵地，以新工科教学科研为主，推进交叉学科研究。北大新工科以“以理启供，以文冶工，以工促多”为建设内涵，近年来依托北大新校区建设提供的新空间和新文化基础，加强重点学科和重点方向机构布局；同时积极探索有效的产学研协同支撑机制，带动基于原始创新和新兴技术的创新创业发展，推动学术链、创新链和产业链高度融合，坚定不移地走稳走好综合性大学发展新工科的特色之路。李水清重点介绍了清华大学加强有组织科研、高质量推进2030创新行动计划的有关工作情况。他表示，清华大学始终把服务国家作为最高追求，把攻克国家亟需解决的重大科技难题作为光荣使命，高质量服务科技自立自强：一是强化问题导向牵引的有组织科研，通过开展2030创新行动计划等，在加强基础研究和关键核心技术攻关、推进产学研深度融合、促进科技开放合作等方面布局；二是完善有组织科研模式，建立资源保障和激励机制，统筹拓展校内外资源，分类按需、精准支持，为重大任务提供教育、科技、人才协同的基础保障；三是坚持开放创新，拓展国际科技合作，牵头或参与国际大科学计划，搭建高水平国际学术交流平台，提升国际科技合作质量。李晓强介绍了北京大学科技创新情况。北京大学贯彻落实“四个面向”战略部署，坚持高水平自由探索，推动高质量有组织科研，不断强化国家战略科技力量，助力实现高水平科技自立自强。他详细介绍了北京大学在推动基础研究、承担重大任务、完善科技创新平台体系、服务重大成果产出、深化国际科技合作和优化政策保障体系方面的部署和成效，展示了北京大学在高质量推动科技创新工作发展和全方位打造国家战略科技力量方面的举措和成绩。刘奕群表示，高水平研究型大学应在人才培养、基础研究、对外合作交流中发挥积极作用。清华大学在这些方面加强布局，积极开展高层次人才培养计划、创新行动计划和全球战略计划，下大力气强化基础研究和开展有组织科研。他建议两校科研管理部门建立协同机制，共同助力国家科技自立自强。谢冰指出，北京大学坚持高水平自由探索和高质量有组织科研相结合，在院系层面积极推动自由探索，同时以基地、平台为抓手强化有组织科研。他建议两校科研管理部门将密切合作交流常态化长效化，积极学习彼此的有益经验，充分发挥各自优势和特色，携手促进高水平科技创新服务高质量发展。刘奕群、谢冰作总结发言此次联学共建交流活动，两校科研管理部门都获益匪浅，既深入学习了习近平新时代中国特色社会主义思想和习近平总书记重要讲话精神，把握高质量发展的深刻内涵，又开展了科研管理服务业务探讨，在加强高水平研究型大学开展有组织科研、服务国家重大需求等方面互相学习借鉴优秀经验和做法。双方表示，今后一定要继续加强交流，深化合作，携手共进，以自信自强、不怕吃苦、勇于担当的昂扬精神，为打造国家战略科技力量、建设科技强国的共同目标不断奋斗。活动合影'
    # text=u'2017年12月12日，为深入学习党的十九大精神，北京大学城市与环境学院2016级硕士生党支部与清华大学工物系研三党支部开展了主题为“青年强则国强 青年兴则国兴”共建交流活动。与会党员结合前期对十九大报告的学习，以“专题分享”“十九大报告知识竞赛”“诗歌朗诵”“主题讨论”及“重温入党誓词”等多种形式进一步增强了对十九大精神和新时代青年人的使命担当的理解和认识。活动合影在“专题分享”环节，北大城环学院的刘飞首先作了题为“深入学习十九大精神 争做又红又专好青年”的主题分享。他结合自身专业，对十九大中指出的“建设生态文明是中华民族永续发展的千年大计，必须树立和践行绿水青山就是金山银山”的理念阐述了自己的心得体会。他与同学们互勉：读万卷书，行万里路，我们要走到祖国各地、走到乡村基层、走到田间地头，去感受中国的国情、社情和民情；发挥自己的专长，运用专业知识解决社会问题，同时将最新、最迫切的问题带回学校研究。努力做到学以致用、产学研紧密结合，更好地服务祖国社会发展。清华大学工物系刘广银作了“盛世中国 奋斗青春”的主题分享。他详细介绍了中国在近五年来取得的辉煌成就，并就“一带一路”建设作了深入介绍。他同时呼吁同学们志存高远、脚踏实地，要将个人梦想与中国梦紧密结合起来，在实践中保持勤奋好学、踏实肯干的态度，一步一个脚印，为中国实现社会主义现代化强国贡献自己的力量。专题分享在“十九大报告知识竞赛”环节，在场的党员被随机分成5组，每组既有北大同学也有清华同学。知识竞赛分为抢答和必答两个环节。整个竞赛过程既激烈又不乏趣味，同学们沉着应对，时而表情凝重、若有所思，时而喜笑颜开、想到答案。经过激烈追逐，最终第一组同学过关斩将，夺得冠军。十九大报告知识竞赛在“诗歌朗诵”环节，北大城环学院张新悦和清华工物系刘志远深情朗诵了《歌颂党》，他们将情感与朗诵融为一体，动情的声线仿似饮泣诉说着对祖国的挚爱，深情歌颂党的声音感染了在场的每一个人，让同学们回顾起祖国的奋斗史，也看到了祖国的美好未来。伴随着时而低沉、时而高亢的朗诵，同学们不时爆发出热烈的掌声。诗歌朗诵在“主题讨论”环节，两个支部同学以“请结合十九大报告中指出的‘青年强则国强 青年兴则国兴’谈谈你认为应该如何培养青年党员的家国情怀与责任担当”“结合网上及身边的舆论，谈谈舆论对社会的影响”“结合支部党建工作经验，谈谈如何更好地推进高校基层党组织建设，从而激发党支部活力，确保学生党支部既能体现学生特色，又能发挥党的基层堡垒作用”和“谈谈你对十九大报告中关于生态文明建设的理解，以及研究生党员在平时的科研学习中应如何推动生态文明建设”为主题进行了深度讨论。有党员谈道，青年党员应加深对中华文化的理解与认同，在此基础上，将家国情怀和责任担当融入到工作学习科研当中去，将自己的发展与国家民族的发展紧密结合在一起。有党员表示在海量信息化的时代，在政府弘扬主旋律、对舆论适当引导的同时，我们学生党员也要有辨别真假舆论的能力；有党员对比了两校党支部开展过的不同活动，总结经验，探索性地提出了贴切学生实际的党建方式。有党员结合十九大报告和自身科研经历，表示研究生党员要将理论学习和科研实践统筹结合起来，既要将科研做到高精尖，又能将科研成果转化，为中国的生态文明建设和绿色发展之路提供科技支撑。主题讨论在“重温入党誓词”环节，党员们全体起立，举起右拳，面向党旗庄严宣誓。党员们一字一句，铿锵有力，坚定不移。在党旗前，他们再一次表明共产党员永远不忘初心，始终牢记使命。重温入党誓词本次支部共建交流进一步丰富了党日活动的形式，“主题分享”加深了支部党员对十九大精神和中国成就的了解；“十九大知识竞赛”让支部党员们对十九大报告了然于心；“诗歌朗诵”让支部党员们回顾祖国历史，展望美好未来；“主题讨论”激发了支部党员们的积极性和创造性；“重温入党誓词”让同学们牢记初心，不忘使命。同时，本次活动也让同学们深深地意识到作为青年党员的责任与担当，纷纷表示要运用所知所学，更好地回馈社会，为中国实现社会主义现代化强国贡献力量。专题链接：学习贯彻十九大精神编辑：山石'

    # print(text)
    text = ' '.join(text)
    token_h = text_to_word_sequence(text)
    test_word_lists.append(token_h)
    # print(test_word_lists)
    # info='预测算法为：'
    out=''



    bilstm_model.model.bilstm.bilstm.flatten_parameters()  # remove warning
    test_word_lists= prepocess_data_for_lstmcrf_test(
        test_word_lists, test=True
    )
    pred_list = bilstm_model.test2(test_word_lists,crf_word2id, crf_tag2id)

    #

    #
    # bilstm_model.model.bilstm.flatten_parameters()  # remove warning
    # # test2直接预测
    # pred_list= bilstm_model.test2(test_word_lists,bilstm_word2id, bilstm_tag2id)

    # print(pred_list)
    for i in range(len(pred_list[0])):
        out_i=test_word_lists[0][i]+" "+pred_list[0][i]+'\n'
        out+=out_i
    # self.out_text.set(out)

    result={
        'DATE':[],
        'SCH': [],
        'COOP': [],
        'CXY': [],
        'JP': []
    }
    i = 0
    while i<len(pred_list[0]):
        #当遇到第一个'-'时，不是'-'时跳出
        while len(pred_list[0][i])>1 :
            added=''
            #最后一个
            # if i==len(pred_list[0]):
            #     if len(pred_list[0][i])>1:
            #         added+=test_word_lists[0][i]
            # else:
            try:
                while len(pred_list[0][i+1])>1:
                    added+=test_word_lists[0][i]
                    i+=1
                    if i == len(pred_list[0])-1:
                        break

                added+=test_word_lists[0][i]
            except IndexError as e:
                print(i)
                print(len(pred_list[0]))
            #下一个为0或最后一个
            if len(pred_list[0][i])>1 and len(added)>1:
                if pred_list[0][i][2:] == 'DATE':
                    result['DATE'].append(added)
                if pred_list[0][i][2:] == 'SCH':
                    result['SCH'].append(added)
                if pred_list[0][i][2:] == 'COOP':
                    result['COOP'].append(added)
                if pred_list[0][i][2:] == 'CXY':
                    result['CXY'].append(added)
                if pred_list[0][i][2:] == 'JP':
                    result['JP'].append(added)
            i+=1
            if i == len(pred_list[0]):
                break
        i+=1

    # print(result)
    results_list.append(result)
    # print(results_list)
print(results_list)
print(len(results_list))

# # 保存为 JSON 文件
# with open('result2.json', 'w') as f:
#     json.dump(results_list, f, indent=4)

#导入结果
workbook = openpyxl.load_workbook(file_path)  # 加载已有文件
# 选择活动的工作表
sheet = workbook.active
# 获取最后一列的索引
last_col = sheet.max_column + 1
sheet.cell(row=1,column=last_col,value='时间')
sheet.cell(row=1,column=last_col+1,value='学校')
sheet.cell(row=1,column=last_col+2,value='合作方')
sheet.cell(row=1,column=last_col+3,value='产学研机构')
sheet.cell(row=1,column=last_col+4,value='挂牌类')
# 将列表中的元素写入最后一列
for i, value in enumerate(results_list, start=2):
    sheet.cell(row=i, column=last_col, value=str(value['DATE']))
    sheet.cell(row=i, column=last_col+1, value=str(value['SCH']))
    sheet.cell(row=i, column=last_col+2, value=str(value['COOP']))
    sheet.cell(row=i, column=last_col+3, value=str(value['CXY']))
    sheet.cell(row=i, column=last_col+4, value=str(value['JP']))


# 保存工作簿
workbook.save(file_path)
    # print(out)
