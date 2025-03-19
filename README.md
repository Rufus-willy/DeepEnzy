<h1 align="center">DeepEnzyme</h1>

本项目对DeepEnzyme（https://github.com/hongzhonglu/DeepEnzyme ）进行了小范围的改动，新增了推理脚本 `run_inference.py`，降低了模型的使用难度。现在，用户可以更便捷地使用本模型进行酶周转数预测。

# Table of contents
- [Table of contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
- [Contributors](#contributors)
- [Citation](#citation)
- [License](#license)
- [Footer](#footer)

# Installation

[(Back to top)](#table-of-contents)

```
conda create -n DeepEnzyme python=3.9
conda activate DeepEnzyme
pip install -r requirements.txt
```

# Usage

[(Back to top)](#table-of-contents)

- Example of DeepEnzyme for turnover number prediction using protein sequence, structure and reaction substrate:
```
Code/Example/example.py
```

### 使用run_inference.py进行推理
1. 确保已经安装所需的依赖库，可通过以下命令安装：
### 使用yml文件安装
1. 创建并激活虚拟环境：
```
conda env create -f env.yml
conda activate deepenzyme
```

2. 运行推理脚本，使用以下命令：
```
python run_inference.py --csv_file <input_csv_file> --output_file <output_csv_file> --fasta_file <fasta_file>
```
其中，`<input_csv_file>` 是包含输入数据的CSV文件路径，`<output_csv_file>` 是输出结果的CSV文件路径，`<fasta_file>` 是包含蛋白质序列的FASTA文件路径。

# Citation

[(Back to top)](#table-of-contents)

- Please cite the paper "[DeepEnzyme: a robust deep learning model for improved enzyme turnover number prediction by utilizing features of protein 3D structures](https://www.biorxiv.org/content/10.1101/2023.12.09.570923v2)"

# Contributors

[(Back to top)](#table-of-contents)

Tong Wang
- State Key Laboratory of Microbial Metabolism, School of Life Science and Biotechnology, Shanghai Jiao Tong University, Shanghai 200240, China
- College of Science, Chongqing University of Technology, Chongqing, 400054, China

Wenbin Su

本修改基于此仓库：https://github.com/hongzhonglu/DeepEnzyme


# License

[(Back to top)](#table-of-contents)

[GNU General Public License version 3](https://opensource.org/licenses/GPL-3.0)

# Footer

[(Back to top)](#table-of-contents)

