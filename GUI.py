import tkinter as tk
import tkutil as tku
from tkinter import filedialog
from utils import load_model, extend_maps, prepocess_data_for_lstmcrf_test
from data import build_corpus
from evaluating import Metrics
from evaluate import ensemble_evaluate
from tensorflow.keras.preprocessing.text import text_to_word_sequence

REMOVE_O = False  # 在评估的时候是否去除O标记

class GUI_start:
	def __init__(self):
		self.root=tk.Tk()
		self.root.geometry("%dx%d" % (500, 300))  # 窗体尺寸
		tku.center_window(self.root)  # 将窗体移动到屏幕中央
		self.root.title("中文实体命名识别系统")
		self.root.resizable(False, False)  # 设置窗体不可改变大小
		self.no_title = True
		self.title = tk.Label(self.root,
							  text='中文实体命名识别系统',
							  font=('Arial', 30),  # 字体和字体大小
							  )
		self.title.pack(side='top')

		self.frame=tk.Frame(self.root)
		self.frame.pack(expand=True)
		# 按钮
		self.btn1 = tk.Button(self.frame,
							  text='训练模型',
							  command=self.func1,
							  width=20,
							  height=3
									  )
		self.btn1.pack(side='left',expand=True)

		self.btn2 = tk.Button(self.frame,
							  text='进行预测',
							  command=self.func2,
							  width=20,
							  height=3
							  )
		self.btn2.pack(side='right',expand=True)

	def func1(self):
		window1=GUI_train()
		window1.root.mainloop()

	def func2(self):
		window2=GUI_pred()
		window2.root.mainloop()



class GUI_pred:
	def __init__(self):
		self.root=tk.Tk()
		self.root.geometry("%dx%d" % (800, 600))  # 窗体尺寸
		tku.center_window(self.root)  # 将窗体移动到屏幕中央
		self.root.title("中文实体命名识别系统")
		self.root.resizable(False, False)  # 设置窗体不可改变大小
		self.no_title = True
		self.title = tk.Label(self.root,
							  text='中文实体命名识别系统',
							  font=('Arial', 30),  # 字体和字体大小
							  )
		self.title.pack(side='top')
		# 选择算法
		self.chooseA = False
		self.chooseB = False
		self.chooseC = False
		self.chooseD = False
		self.v = tk.IntVar()
		self.title2 = tk.Label(self.root,
							   text='请选择预测算法',
							   font=('Arial', 18),  # 字体和字体大小
							   )
		self.title2.pack(side='top', anchor='sw')

		self.frame_model = tk.Frame(self.root)
		self.frame_model.pack(side='top', anchor='sw')
		self.radioBtnA = tk.Radiobutton(self.frame_model,
										text="HMM",
										variable=self.v,
										value=1,
										command=self.funChooseHMM)
		self.radioBtnB = tk.Radiobutton(self.frame_model,
										text="CRF",
										variable=self.v,
										value=2,
										command=self.funChooseCRF)
		self.radioBtnC = tk.Radiobutton(self.frame_model,
										text="BiLSTM",
										variable=self.v,
										value=3,
										command=self.funChooseBiLSTM)
		self.radioBtnD = tk.Radiobutton(self.frame_model,
										text="CRF+BiLSTM",
										variable=self.v,
										value=4,
										command=self.funChooseCRF_BiLSTM)
		self.radioBtnA.pack(side='left')
		self.radioBtnB.pack(side='left')
		self.radioBtnC.pack(side='left')
		self.radioBtnD.pack(side='left')
		# 选择模型 self.model_path
		self.model_frame = tk.Frame(self.root)
		self.model_frame.pack(side='top', anchor='sw')
		self.model_label = tk.Label(self.model_frame, text='请导入模型：', font=('华文彩云', 18))
		self.model_label.pack(side='left')
		self.model_path = tk.StringVar()
		self.model_entry = tk.Entry(self.model_frame, textvariable=self.model_path, font=('FangSong', 10), width=50,
							   state='readonly')
		self.model_entry.pack(side='left')
		self.model_btn = tk.Button(self.model_frame,
									  text='选择路径',
									  command=self.choose_model_path
									  )
		self.model_btn.pack(side='left')
		# 输入文本 self.text
		self.text = tk.StringVar()
		self.text_frame = tk.Frame(self.root)
		self.text_frame.pack(side='top', anchor='sw')
		self.new_frame=tk.Frame(self.text_frame)
		self.new_frame.pack(side='left',anchor='nw')
		self.text_label = tk.Label(self.new_frame, text='请输入预测文本：', font=('华文彩云', 18))
		self.text_label.pack(side='left')
		self.text_entry=tk.Entry(self.text_frame, textvariable=self.text, font=('FangSong', 13), width=50,
							   state='normal')
		self.text_entry.pack(side='left',expand=True,ipadx=20,ipady=50)
		# 预测按钮
		self.pred_frame = tk.Frame(self.text_frame)
		self.pred_frame.pack(side='top', anchor='sw')
		self.pred_btn=tk.Button(self.pred_frame,
									  text='进行预测',
									  command=self.pred,
								width=15,height=2
									  )
		self.pred_btn.pack(side='left')
		# self.label = tk.Label(self.frame, text='验证数据集：', font=('华文彩云', 15))
		# self.label.pack(side='left')
		# self.text = tk.StringVar()
		# self.entry = tk.Entry(self.frame, textvariable=self.text, font=('FangSong', 10), width=50,
		# 					  state='readonly')
		# self.entry.pack(side='left')
		# self.dev_data_btn = tk.Button(self.frame,
		# 							  text='选择路径',
		# 							  command=self.
		# 							  )
		# self.dev_data_btn.pack(side='left')
		# 结果输出
		self.out_text = tk.StringVar()
		self.out_frame=tk.Frame(self.root)
		self.out_frame.pack(side='top', anchor='sw')
		self.out_label_frame=tk.Frame(self.out_frame)
		self.out_label_frame.pack(side='left',anchor='nw')
		self.out_label = tk.Label(self.out_label_frame, text='预测结果：', font=('华文彩云', 18))
		self.out_label.pack(side='left') #label
		# self.out_entry=tk.Label(self.out_frame, textvariable=self.out_text, width=60,
		# 					   state='disabled',bd=2.5)
		self.out_entry = tk.Text(self.out_frame, width=50,height=150,font=('FangSong', 13),
								  state='normal')
		self.scroll = tk.Scrollbar()
		self.scroll.pack(side='left',fill='y')
		self.scroll.config(command=self.out_entry.yview)
		self.out_entry.config(yscrollcommand=self.scroll.set)
		self.out_entry.pack(side='left',expand=True,ipadx=60,ipady=150,pady=10)

	def pred(self):
		train_word_lists=[]
		train_tag_lists=[]
		word2id={}
		tag2id={}
		bilstm_word2id={}
		bilstm_tag2id={}
		crf_word2id={}
		crf_tag2id={}
		self.out_entry.delete(1.0, "end")
		print("预测模型地址为" + self.model_path.get())
		print("预测文本为"+self.text.get())
		# 导入word2id, tag2id
		train_word_lists, train_tag_lists, word2id, tag2id = \
			build_corpus("train")
		# 构造test_word_list
		test_word_lists=[]
		text = self.text.get()
		print(text)
		text = ' '.join(text)
		token_h = text_to_word_sequence(text)
		test_word_lists.append(token_h)
		# print(test_word_lists)
		info='预测算法为：'
		out=''

		if self.chooseA==True:
			info+='HMM'
			hmm_model = load_model(self.model_path.get())
			pred_list=hmm_model.test(test_word_lists,
                              word2id,
                              tag2id)

		elif self.chooseB==True:
			info+='CRF'
			crf_model = load_model(self.model_path.get())
			pred_list = crf_model.test(test_word_lists)

		elif self.chooseC==True:
			info += 'BiLSTM'
			bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
			bilstm_model = load_model(self.model_path.get())
			bilstm_model.model.bilstm.flatten_parameters()  # remove warning
			# test2直接预测
			pred_list= bilstm_model.test2(test_word_lists,bilstm_word2id, bilstm_tag2id)

		elif self.chooseD == True:
			info += 'BiLSTM+CRF'
			crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
			bilstm_model = load_model(self.model_path.get())
			bilstm_model.model.bilstm.bilstm.flatten_parameters()  # remove warning
			test_word_lists= prepocess_data_for_lstmcrf_test(
				test_word_lists, test=True
			)
			pred_list = bilstm_model.test2(test_word_lists,crf_word2id, crf_tag2id)


		print(pred_list)
		for i in range(len(pred_list[0])):
			out_i=test_word_lists[0][i]+" "+pred_list[0][i]+'\n'
			out+=out_i
		# self.out_text.set(out)
		# print(out)
		info+="\n预测模型地址为" + self.model_path.get()+"\n"
		self.out_entry.insert(tk.INSERT,info)
		self.out_entry.insert(tk.INSERT,out)


	# 按钮选择算法
	def funChooseHMM(self):
		if self.chooseA == False:
			self.chooseA = True
			self.chooseB == False
			self.chooseC == False
			self.chooseD == False
		else:
			self.chooseA = False

	def funChooseCRF(self):
		if self.chooseB == False:
			self.chooseB = True
			self.chooseA = False
			self.chooseC == False
			self.chooseD == False
		else:
			self.chooseB = False

	def funChooseBiLSTM(self):
		if self.chooseC == False:
			self.chooseC = True
			self.chooseA = False
			self.chooseB == False
			self.chooseD == False
		else:
			self.chooseC = False


	def funChooseCRF_BiLSTM(self):
		if self.chooseD == False:
			self.chooseD = True
			self.chooseA = False
			self.chooseB == False
			self.chooseC == False
		else:
			self.chooseD = False

	# 选择模型路径
	def choose_model_path(self):
		path = filedialog.askopenfilename(title='请选择文件')
		self.model_path.set(path)


class GUI_train:
	def __init__(self):
		self.root = tk.Tk()
		self.root.geometry("%dx%d" % (1000, 800))   # 窗体尺寸
		tku.center_window(self.root)               # 将窗体移动到屏幕中央
		self.root.title("中文实体命名识别系统")
		self.root.resizable(False, False)          # 设置窗体不可改变大小
		self.no_title = True
		#title
		self.title=tk.Label(self.root,
							text='中文实体命名识别系统',
							font=('Arial', 30),     #字体和字体大小
							)
		self.title.pack(side='top')

		self.title = tk.Label(self.root,
							  text='数据集',
							  font=('Arial', 20),  # 字体和字体大小
							  )
		self.title.pack(side='top',anchor='sw')

		# train_data
		self.frame_data_1=tk.Frame(self.root)
		self.frame_data_1.pack(side='top',anchor='sw')
		self.label1 = tk.Label(self.frame_data_1, text='训练数据集：', font=('华文彩云', 15))
		self.label1.pack(side='left')

		self.text1=tk.StringVar()
		self.entry1= tk.Entry(self.frame_data_1, textvariable=self.text1, font=('FangSong', 10), width=50, state='readonly')
		self.entry1.pack(side='left')

		self.train_data_btn=tk.Button(self.frame_data_1,
								 text='选择路径',
								 command=self.choose_train_path
								 )
		self.train_data_btn.pack(side='left')

		#dev_data
		self.frame_data_2 = tk.Frame(self.root)
		self.frame_data_2.pack(side='top', anchor='sw')
		self.label2 = tk.Label(self.frame_data_2, text='验证数据集：', font=('华文彩云', 15))
		self.label2.pack(side='left')
		self.text2 = tk.StringVar()
		self.entry2 = tk.Entry(self.frame_data_2, textvariable=self.text2, font=('FangSong', 10), width=50, state='readonly')
		self.entry2.pack(side='left')
		self.dev_data_btn = tk.Button(self.frame_data_2,
										text='选择路径',
										command=self.choose_dev_path
										)
		self.dev_data_btn.pack(side='left')

		# test——data
		self.frame_data_3 = tk.Frame(self.root)
		self.frame_data_3.pack(side='top', anchor='sw')
		self.label3 = tk.Label(self.frame_data_3, text='测试数据集：', font=('华文彩云', 15))
		self.label3.pack(side='left')
		self.text3 = tk.StringVar()
		self.entry3 = tk.Entry(self.frame_data_3, textvariable=self.text3, font=('FangSong', 10), width=50, state='readonly')
		self.entry3.pack(side='left')
		self.test_data_btn = tk.Button(self.frame_data_3,
										text='选择路径',
										command=self.choose_test_path
										)
		self.test_data_btn.pack(side='left')

		# 算法
		self.chooseA=False
		self.chooseB = False
		self.chooseC = False
		self.chooseD = False
		self.v=tk.IntVar()
		self.title2 = tk.Label(self.root,
							  text='模型',
							  font=('Arial', 20),  # 字体和字体大小
							  )
		self.title2.pack(side='top', anchor='sw')
		# 算法
		self.frame_model=tk.Frame(self.root)
		self.frame_model.pack(side='top',anchor='sw')
		self.radioBtnA = tk.Radiobutton(self.frame_model,
								   text="HMM",
								   variable=self.v,
								   value=1,
								   command=self.funChooseHMM)
		self.radioBtnB = tk.Radiobutton(self.frame_model,
										text="CRF",
										variable=self.v,
										value=2,
										command=self.funChooseCRF)
		self.radioBtnC = tk.Radiobutton(self.frame_model,
										text="BiLSTM",
										variable=self.v,
										value=3,
										command=self.funChooseBiLSTM)
		self.radioBtnD = tk.Radiobutton(self.frame_model,
										text="CRF+BiLSTM",
										variable=self.v,
										value=4,
										command=self.funChooseCRF_BiLSTM)
		self.radioBtnA.pack(side='left')
		self.radioBtnB.pack(side='left')
		self.radioBtnC.pack(side='left')
		self.radioBtnD.pack(side='left')
		# 训练模型按钮
		self.train_btn_frame = tk.Frame(self.root)
		self.train_btn_frame.pack(side='top', anchor='nw')
		self.train_btn = tk.Button(self.train_btn_frame,
								  text='训练模型',
								  command=self.train,
								  width=15, height=2
								  )
		self.train_btn.pack()

		# 结果输出 修改即可self.out_entry
		self.out_text = tk.StringVar()
		self.out_frame = tk.Frame(self.root)
		self.out_frame.pack(side='top', anchor='sw')
		self.out_label_frame = tk.Frame(self.out_frame)
		self.out_label_frame.pack(side='left', anchor='nw')
		self.out_label = tk.Label(self.out_label_frame, text='训练结果：', font=('华文彩云', 18))
		self.out_label.pack(side='left')  # label
		# self.out_entry=tk.Label(self.out_frame, textvariable=self.out_text, width=60,
		# 					   state='disabled',bd=2.5)
		self.out_entry = tk.Text(self.out_frame, width=50, height=150, font=('FangSong', 13),
								 state='normal')
		self.scroll = tk.Scrollbar()
		self.scroll.pack(side='left', fill='y')
		self.scroll.config(command=self.out_entry.yview)
		self.out_entry.config(yscrollcommand=self.scroll.set)
		self.out_entry.pack(side='left', expand=True, ipadx=60, ipady=150, pady=10)

	def train(self):
		train_word_lists, train_tag_lists, word2id, tag2id = \
			build_corpus("train")
		dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
		test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)
		if self.chooseA == True:

			pass
		elif self.chooseB == True:
			pass
		elif self.chooseC == True:
			pass
		elif self.chooseD == True:
			pass


	# 选择路径
	def choose_train_path(self):
		"""注意，以下列出的方法都是返回字符串而不是数据流"""
		# 返回一个字符串，且只能获取文件夹路径，不能获取文件的路径。
		# path = filedialog.askdirectory(title='请选择一个目录')

		# 返回一个字符串，可以获取到任意文件的路径。
		path = filedialog.askopenfilename(title='请选择文件')
		self.text1.set(path)

	def choose_dev_path(self):
		path = filedialog.askopenfilename(title='请选择文件')
		self.text2.set(path)

	def choose_test_path(self):

		path = filedialog.askopenfilename(title='请选择文件')
		self.text3.set(path)

	# 选择模型
	def funChooseHMM(self):
		if self.chooseA==False:
			self.chooseA=True
		else:
			self.chooseA=False
	def funChooseCRF(self):
		if self.chooseB==False:
			self.chooseB=True
		else:
			self.chooseB=False
	def funChooseBiLSTM(self):
		if self.chooseC==False:
			self.chooseC=True
		else:
			self.chooseC=False
	def funChooseCRF_BiLSTM(self):
		if self.chooseD==False:
			self.chooseD=True
		else:
			self.chooseD=False



if __name__ == "__main__":
	app = GUI_pred()
	# app=GUI_train()
	app.root.mainloop()
