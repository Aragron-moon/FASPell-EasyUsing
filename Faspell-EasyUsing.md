# Faspell-EasyUsing

本项目为了帮助大家更快更简单的实现对Faspell的应用，在阅读论文《FASPell: A Fast, Adaptable, Simple, Powerful Chinese Spell Checker Based On DAE-Decoder Paradigm》的基础上，结合爱奇艺提供的源代码进行模型实现应用。该项目仅仅对源代码的关键地方做出***中文注释***，同时提供更详细的***训练方式解读***，以及一组***可直接应用***的模型过滤数据。论文链接和Faspell源码链接如下：

[论文](https://www.aclweb.org/anthology/D19-5522.pdf)

[源码](https://github.com/iqiyi/FASPell)

特别感谢该篇论文和源码贡献的作者之一：**洪煜中**前辈，感谢前辈在这个过程中提供的帮助！

------

该说明文档将从几个方面进行阐述：

+ 源码目录结构和该项目目录结构对比
+ 模型训练过程的详细介绍
+ 总结

------

## 源码目录结构和该项目目录结构对比

![](https://github.com/Aragron-moon/FASPell-EasyUsing/blob/master/images/projectStruct.png)

下载好Faspell并用pycharm打开的目录结构如上：

+ bert_modified：主要是对原Bert模型进一步作微调的子目录
  + create_data.py   对原始数据进行处理
  + create_tf_record.py   根据处理后的数据生成满足微调要求的tf_record数据
  + modeling.py   Bert模型的源代码
  + tokenization.py   对文本进行tokenize
+ data：主要存放模型训练和使用过程中的相关数据
  + char_meta.txt   对单个字到发音和字形拆解映射的示例
  + ocr_test_1000.txt   用来进行测试的OCR数据集
  + ocr_train_3575.txt   用来进行训练的OCR数据集
+ char_sim.py ：用来计算字音和字形相似度
+ faspell.py：模型训练和应用
+ faspell_configs.json：模型训练和应用的配置
+ masked_lm.py：对原Bert作修改后的MLM
+ model_fig.png：模型训练的图示
+ plot.py：训练过程中的结构图绘制和相关数据生成
+ model：模型的参数保存文件
  + fine-tuned：微调的模型参数
  + pre-trained：原Bert的模型参数

按照原项目的说明，需要手动新建model文件和该文件下的两个子文件fine-tuned、pre-trained用来存储模型参数

该项目在pycharm上的目录结构如下：

![](https://github.com/Aragron-moon/FASPell-EasyUsing/blob/master/images/newProjectStruct.png)

两个项目结构不一样的地方（只介绍重要的地方）：

+ bert_modified：引入了模型微调需要用到的代码，均是直接从Google上进行copy
  + optimization.py   模型微调过程中要使用的优化器
  + run_pretraining.py   模型微调的代码
+ data：引入了模型在训练和应用过程的数据
  + all_char_meta.txt   所有字到字音和字形结构的映射（U+4E5F  也  ye3,yi2;jaa5;YA;YA;dã  ⿻乜丨）
  + ALLSIGHAN.txt   SIGHAN13-15所有训练集和数据集的汇总并做了繁转简（2   敬祝身体建慷。    敬祝身体健康。）
  + ALLSIGHAN-15.txt   上述文件中去除了SIGHAN15中的训练集和测试集
  + ids.txt   原项目所提到的字到字形映射（iU+4E27  丧  ⿱⿻土丷⿰𠄌⿺乀丿）
  + idsCharge.txt   对上述文件进行人工丰富
  + test_ofsighan15.txt   用于测试的SIGHAN15测试集
  + train_srcofsighan15.txt   用于训练的SIGHAN15训练集
  + Unihan_Readings.txt   原项目所提到的字到字音的映射（U+3402  kCantonese hei2）

该项目也在原项目的基础上，对一些代码做了修改，所有的修改均在代码文件中做了明确的注释。

----

## 模型训练过程的详细介绍

对于Faspell的训练分为三个过程：**预训练**，**微调**，**训练CSD**

**预训练**

这一部分其实不需要进行，我们直接下载Google预训练好的模型就可以了，下载路径如下：

[Bert预训练模型]()   也可以在原项目中直接获取到下载路径，并将下载好的文件放到model/pre-trained/下,文件结构如下所示：

![](https://github.com/Aragron-moon/FASPell-EasyUsing/blob/master/images/pre-trained.png)

**微调**

**Step1**

在bert_modified目录下，运行create_data.py

`python create_data.py -f '../data/SIGHAN-15.txt'`

必须指定的参数：

输入文件路径   -f '../data/SIGHAN-15.txt'   指定原始数据；

其余参数根据需要自行指定（基本不用）。

该命令执行完后，会在bert_modified目录下生成两个新文件wrong.txt和correct.txt。

**Step2**

在bert_modified目录下，运行create_tf_record.py

`python create_tf_record.py --input_file correct.txt --wrong_input_file wrong.txt --output_file tf_examples.tfrecord --vocab_file ../model/pre-trained/vocab.txt`

必须指定的参数：

输入文件   --input_file correct.txt   为上一命令的生成结果；

​                  --wrong_input_file wrong.txt

输出文件   --output_file tf_examples.tfrecord   指定输出文件；

词表文件   --vocab_file ../model/pre-trained/vocab.txt   指定pre-trained下的vocab.txt；

其余参数根据需要自行指定（基本不用）。

该命令执行完后，会在bert_modified目录下生成tf_examples.tfrecord。

**Step3**

在bert_modified目录下，运行run_pretraining.py

`python run_pretraining.py --input_file './tf_examples.tfrecord' --bert_config_file '../model/pre-trained/bert_config.json' --output_dir '../model/fine-tuned/' --init_checkpoint '../model/pre-trained/bert_model.ckpt' --do_train True`

必须指定的参数：

输入文件   --input_file './tf_examples1.tfrecord'   为上一命令的生成结果；

Bert的配置文件   --bert_config_file '../model/pre-trained/bert_config.json'   指定为pre-trained下的配置文件；

模型微调参数的保存位置   --output_dir '../model/fine-tuned/'   指定为modl/fine-tuned/；

模型初始化参数的保存位置   --init_checkpoint '../model/pre-trained/bert_model.ckpt'   指定为pre-trained下的参数；

指定是否为训练模式   --do_train True；

其余参数根据需要自行指定（基本不用）。

需要注意：

+ 模型在训练过程中，由于显卡的不一致会因为batchsize的值而导致OOM错误，解决该问题可以通过更改run_pretraining.py中batchsize的默认值或者在执行命令时通过参数train_batch_size设置batchsize的值（我已经在run_pretraining.py中修改了batchsize默认值为16）；
+ 模型在微调时，指定训练Step为10000步时停止（论文中给出的具体训练步数），对于这一点其实是有待商榷的，具体原因会在总结中说明。在模型微调时，可以在model/fine-tuned/下观察到不断有新的模型参数被保存，当10000步模型参数保存后，就可以停止模型的训练；

**训练CSD**

我认为这一部分才是Faspell的精华和重点所在，在进行CSD的具体训练步骤开始之前，先详细的介绍以下文件：

faspell_configs.json：

+ general_configs：训练的通用配置
  + rank：rank的值设定，该值得具体含义可以在论文中找到。简单点就是每个位置对应的候选集中排第几的候选值（如rank=2，就是指在每个位置对应token的前几个候选值中排第2的那一个候选值）
  + round：模型重复生成的次数。（如round=2，模型会将第一次的输出结果当做输入再丢到模型中，得到的输出则为最终的结果）
  + weights：模型在计算哪一种相似度，仅在训练时有效。（如visual：1，phonological：0则为字形相似度；visual：0，phonological：1则为字音相似度）
+ char_meta：指定字到字形和字音结构的映射文件
+ lm：MLM的相关配置
  + max_seq：模型允许的最常Seq
  + top_n：模型生成中候选集的大小，即对每一个位置采用前top_n个候选值组成候选集
  + batch_size：一次进入模型的Seq数
  + vocab：指定模型的词表文件
  + bert_configs：指定模型的配置参数
  + pre-trained：指定模型预训练参数位置
  + fine-tuned：指定模型微调参数位置
  + fine_tuning_is_on：指定模型是否使用微调后的参数
+ exp_configs：实验的相关配置
  + union_of_sim：是否计算联合相似度（在训练时指定为false，应用时指定为true）
  + testing_set：模型的测试集
  + training_set：模型的训练集
  + tackle_n_gram_bias：是否进行n_gram_bias（在代码中给出了相应注释，解释了其作用）
  + dump_candidates：存储模型生成结果的文件（训练时，必须指定！因为不指定该文件，那么模型在每一次训练时都会对训练集进行结果生成，这显然会造成大量时间的浪费；可以根据自己的需要进行指定，我所指定的文件是'./pre_train.txt'；应用时，不指定任何值）
  + read_from_dump：是否从保存结果中读取（第一次训练时指定为false，后面的训练指定为true，应用时指定为false）
  + filter_is_on：是否开启过滤
  + store_plots：训练结果图的存储路径（根据需要自行指定；我指定的是‘plots’，所以需要在根目录下新建文件夹plots）
  + store_latex：训练结果图中所有点的详细信息存储路径（根据需要自行指定；我指定的是‘./’）

下面开始进行CSD的训练步骤：

**Step1**

数据准备，对ids.txt进行人工丰富

之所以要对按照原项目的提示下载下来的ids.txt进行进一步处理，是因为在原来的ids.txt中仍然存在大量的可以拆解的字，比如：U+725B  牛  牛；我们把这种不能拆解的字归纳为基本结构，但是ids.txt中存在着很对类似于“牛”这样的仍然可以拆解的假基本结构，所以我们需要将这些仍然可以拆解的字找出来，进行人工拆解。在对这些假基本结构字做了处理后，生成新的字到字形结构的映射文件all_char_meta.txt（我对其中的150多个假基本结构进行了拆分，大家根据需要可以自己进行处理）。理论上，所拆分的字形粒度越细越好。

对于字形结构的具体拆分，大家观察几个ids.txt中的示例就能弄明白，此处不做详细介绍，也可以直接采用该项目中提供的all_char_meta.txt文件。

**Step2**

开始训练CSD，这其实是一个细致活，本身没有什么技巧难度而言，只是要做到仔细。在开始之前，需要检查faspell.py文件中的部分内容是否如下所示：

![](https://github.com/Aragron-moon/FASPell-EasyUsing/blob/master/images/train_before.png)

确定每一个地方都是`Curves.curve_null`，因为我们训练的目的就是为了找打合适的过滤曲线，所以必须确保得到所有的原始数据，这里的函数就是什么都不过滤，当我们训练结束和，就可以将找到的过滤去曲线进行替换。

根据训练的不同参数指定，会产生2 x max_rank x 2种训练结果：第一个2是因为根据候选集中第一个候选值是否和原字符一致将结果分为了2类；max_rank 表示训练的最大rank数；第二个2是因为针对字音相似（P）和字形相似（S）分为两种情况。当max_rank 设置为3的时候，会有以下12种实验结果：

+ **difference=1,rank=0,S**

  在运行命令之前，设置faspell_configs.json中：

  rank为0；

  round为1；

  weights中（visual:1,pronological:0）；

  char_meta为data/all_char_meta.txt；

  max_seq为128；

  top_n为5；

  batch_size为5；

  vocab为model/pre-trained/vocab.txt；

  bert_configs为model/pre-trained/bert_config.json；

  pre-trained为model/pre-trained/bert_model.ckpt；

  fine-tuned为model/fine-tune-15/model.ckpt-12000；

  fine_tuning_is_on为true；

  union_of_sims为false；

  testing_set为./data/test_ofsighan15.txt；

  training_set为./data/train_srcofsighan15.txt；

  tackle_n_gram_bias为true；

  dump_candidates为 ./pre-train.txt；

  read_from_dump为false;

  filter_is_on为true;

  store_plots为plots;

  store_latex为./;

  对于上述配置，除了rank为0以及weights的指定，其余的都是我的实验配置，需要根据自己实际对每一个参数进行指定，每一个参数的意义在前面给了说明（可以直接就按照我这个实验配置进行）

在根目录下运行faspell.py

`python faspell.py -m e -t -d`

必要参数说明：

指定运行模式   -m e   指定为实验模式  -t   指定为训练模式   -d   指定difference为1（候选集中的第一个候选值和原字符不一致）

运行该命令结束后，会在根目录下生成以下几个文件

**pre-train.txt**：保存MLM的生成结果文件

**history.json**   本次训练的历史记录

eg：

 {
        "original_sentence": "但是我不能去参加，因为我有一点事情阿！",
        "correction_history": [
            "但是我不能去参加，因为我有一点事情啊！"
        ]
    }

**dirrerence_1-rank_0-results_0.json**：本次训练对训练集中的每个句子的result

eg:

 {
        "original_sentence": "但是我不能去参加，因为我有一点事情阿！",
        "corrected_sentence": "但是我不能去参加，因为我有一点事情啊！",
        "num_errors": 1,
        "errors": [
            {
                "error_position": 17,
                "original": "阿",
                "corrected_to": "啊",
                "candidates": {
                    "啊": 0.9997532963752747,
                    "阿": 0.00017453855252824724,
                    "！": 2.7026138923247345e-05,
                    "呀": 1.5083395737747196e-05,
                    "喔": 1.141958091466222e-05
                },
                "confidence": 0.9997532963752747,
                "similarity": 0.7777777777777778,
                "sentence_len": 19
            }
        ]
    }

difference_1-rank_0-results_0.png&&difference_1-rank_0-results_02.png：本次训练结果的所有错误点的分布，在根目录下plots文件夹下

eg：

![](https://github.com/Aragron-moon/FASPell-EasyUsing/blob/master/images/difference_1-rank_0-results_0.png)

![](https://github.com/Aragron-moon/FASPell-EasyUsing/blob/master/images/difference_1-rank_0-results_02.png)

difference_1-rank_0-results_0.txt：本次训练的结果

eg：

1	听起来是一份很好的公司。又意思又很多钱。	听起来是一份很好的公司。有意思又很多钱。

difference_1-rank_0-results_0_falsely_corrected.txt：本次训练错误纠正的字

eg：

我要先说你恭喜的话，应为我者到现在很难找到工作，所以很棒。	pos=13	w=者	c=知	p=们	sim=0.0	absent=yes

difference_1-rank_0-results_0_falsely_detected.txt：本次训练错误检查的字

eg：

听说你准备开一个祝会，那天我不能跟你参加。我很抱见阿。	pos=8	w=祝	c=祝	p=晚	sim=0.3846153846153846	absent=yes

difference_1-rank-0-results_0_latex.txt：本次训练过程中绘制图像的所有点的坐标信息

eg：
(0.9997532963752747,0.7777777777777778)[a]

+ **difference=1,rank=1,S**

  在运行命令之前，需要将faspell_configs.json中的

  rank改为1；

  read_from_dump该为true;

在根目录下运行faspell.py

`python faspell.py -m e -t -d`

+ **difference=1,rank=2,S**

  在运行命令之前，需要将faspell_configs.json中的

  rank改为2；

在根目录下运行faspell.py

`python faspell.py -m e -t -d`

+ **difference=1,rank=3,S**

  在运行命令之前，需要将faspell_configs.json中的

  rank改为3；

在根目录下运行faspell.py

`python faspell.py -m e -t -d`

+ **difference=1,rank=0,P**

  在运行命令之前，需要将faspell_configs.json中的

  rank改为0；

  weights改为（visual:0,pronological:1）；

在根目录下运行faspell.py

`python faspell.py -m e -t -d`

+ **difference=1,rank=1,P**

  在运行命令之前，需要将faspell_configs.json中的

  rank改为1；

在根目录下运行faspell.py

`python faspell.py -m e -t -d`

+ **difference=2,rank=0,P**

  在运行命令之前，需要将faspell_configs.json中的

  rank改为2；

在根目录下运行faspell.py

`python faspell.py -m e -t -d`

+ **difference=3,rank=0,P**

  在运行命令之前，需要将faspell_configs.json中的

  rank改为3；

在根目录下运行faspell.py

`python faspell.py -m e -t -d`

后面剩下的6组实验，和前面的6组实验很相似，唯一的不同点是在执行的所有命令当中去掉了参数-d

针对12组训练的实验结果图，进行观察，对每一组结果图（difference_1-rank_0-results_0.png&&difference_1-rank_0-results_02.png）人工得到过滤曲线（效果如下，图片来自论文）

![](https://github.com/Aragron-moon/FASPell-EasyUsing/blob/master/images/picture.png)

对于曲线的实现，采用条曲线的切线进行实现，如代码中给出的示例(采用两条直线来模拟曲线)：

`    def curve_01(confidence, similarity):
        flag1 = 20 / 3 * confidence + similarity - 21.2 / 3 > 0
        flag2 = 0.1 * confidence + similarity - 0.6 > 0
        if flag1 or flag2:
            return True
        return False`

这样，对于每一组结果都能确定一种过滤方式，将过滤方式按照上述代码写好后，添加到对应位置即可

比如，确定了difference=0，rank=1，S的过滤函数如下:

`    def curve_d0r1s(confidence, similarity):
        flag1 = 1 * confidence + similarity * 1 -1> 0
        flag2 = similarity > 0.4
        if flag1 or flag2:
            return True
        return False`

在对应的位置进行更改（如果改组训练结果图片中为0，则代码更改为curve_full）：

训练前：

![](https://github.com/Aragron-moon/FASPell-EasyUsing/blob/master/images/train_before1.png)

训练后，得到过滤方式，进行更改：

![](https://github.com/Aragron-moon/FASPell-EasyUsing/blob/master/images/train_after.png)

**我在该项目的代码中给出了一些例子，这些例子没有严格得出，只是为了更好的理解代码。所有的过滤曲线都需要自己通过做实验的方式得出！**

将12组训练方式得到的过滤曲线替换完成后，CSD的训练也结束了，下面就是应用，在应用模型前，需要将faspell_configs.json的部分配置信息做更改

union_of_sims改为true；

dump_candidates设为空；

read_from_dump改为false；

***Faspell应用***

提供3种应用方式

s：命令行方式，在项目根目录运行命令

`python faspell.py 扫吗关注么众号 受奇艺全网首播 -m s`

f：对文件纠错，在项目根目录下运行命令（其中待纠错文件每一行为一条待纠错句子）

`python faspell.py -m f -f '/data/待纠错文件'`

e：实验模式，在项目根目录下运行命令（对faspell_configs.json中指定的测试文件进行纠错）

`python faspell.py -m e`

**所有结果都会以文件的形式返回在根目下！**



通过实验，在SIGHAN15的测试集上检错和纠错的A、P、R、F值分别为61.4,56.6,53.2,54.8；58.5,47.1,44.4,45.7【ids.txt的字形拆解粒度，微调时的bs和step，训练CSD时过滤曲线的选择是否合理都会影响结果数据】

-----

## 总结

Faspell最大的亮点在于拜托了传统的纠错对ConfusionSet的依赖，采用另外一种思路引入了混淆集的应用。

该思路旨在每一个位置给定的候选集中，从字音和字形同原字的相似度方面考虑，选择最合适的目标字，也正是因为这样，就可以解释Faspell为什么速度会那么快的原因。正如论文中所举的例子，在句子“国际电台苦名丰持人”纠正为“国际电台著名主持人”，对于“苦”不选择排名第一的“知”，结合相似度而选择排名第二的“著”。

但这也体现了Faspell的另一弊端：

+ 当正确答案不在候选集中，模型是不可能纠正正确的。换句话说，微调的MLM给出了Faspell的上限；
+ 因为对于最终的过滤曲线需要通过人工寻找，在这个地方投入的人力成本其实非常大；
+ 对于MLM的微调，因为数据集不够大的问题，batchsize和训练步数的设定都是需要考量的地方，通过实验观察，当MLM在训练集上迭代次数够多的时候，单MLM本身在测试集上的纠错表现就会明显突出；当MLM在训练集上迭代次数不够的时候，单MLM本身就存在很多正确答案无法出现在候选集中，导致Faspell的上限降低。所以对于模型在微调时的bs和step该如何设定，需要试验观察。这里论文中提到采用仅仅从SIGHAN13-15中所有数据集剔除SIGHAN15的测试集作为训练集进行模型的训练和微调，我认为欠妥了，所以我在试验是采用的训练集只是用了SIGHAN13-14的全部数据，这一点可以根据自己的需要进行设置。

***长路漫漫，仍需远行***
