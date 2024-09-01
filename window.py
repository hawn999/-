import tkinter as tk
import tkutil as tku
from tkinter import filedialog
from utils import load_model, extend_maps, prepocess_data_for_lstmcrf
from data import build_corpus
from evaluating import Metrics
from evaluate import ensemble_evaluate

HMM_MODEL_PATH = './ckpts/hmm.pkl'
CRF_MODEL_PATH = './ckpts/crf.pkl'
BiLSTM_MODEL_PATH = 'ckpts/old_bilstm.pkl'
BiLSTMCRF_MODEL_PATH = 'ckpts/old_bilstm_crf.pkl'

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
		self.text_frame = tk.Frame(self.root)
		self.text_frame.pack(side='top', anchor='sw')
		self.new_frame=tk.Frame(self.text_frame)
		self.new_frame.pack(side='left',anchor='nw')
		self.text_label = tk.Label(self.new_frame, text='请输入预测文本：', font=('华文彩云', 18))
		self.text_label.pack(side='left')
		self.text=tk.StringVar()
		self.text_entry=tk.Entry(self.text_frame, textvariable=self.text, font=('FangSong', 10), width=50,
							   state='normal')
		self.text_entry.pack(side='left',expand=True,ipadx=60,ipady=50)
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

	def pred(self):
		print("预测模型地址为" + self.model_path.get())
		print("预测文本为"+self.text.get())
		# 导入word2id, tag2id
		train_word_lists, train_tag_lists, word2id, tag2id = \
			build_corpus("train")
		if self.chooseA==True:
			pass
		elif self.chooseB==True:
			pass
		elif self.chooseB==True:
			pass
		elif self.chooseB == True:
			pass



	# 选择算法
	def funChooseHMM(self):
		if self.chooseA == False:
			self.chooseA = True
		else:
			self.chooseA = False

	def funChooseCRF(self):
		if self.chooseB == False:
			self.chooseB = True
		else:
			self.chooseB = False

	def funChooseBiLSTM(self):
		if self.chooseC == False:
			self.chooseC = True
		else:
			self.chooseC = False

	def funChooseCRF_BiLSTM(self):
		if self.chooseD == False:
			self.chooseD = True
		else:
			self.chooseD = False

	# 选择模型
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

		# 模型
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
	app.root.mainloop()

# window = tk.Tk()
# window.title('中文实体命名识别系统')
# window.geometry('800x800')

# 这里是窗口的内容


# window.mainloop()