# 敏感性分析曲线
import numpy as np
import matplotlib.pyplot as plt

def polynomial_model(x1, x2, x3, x4, x5, x6):
    # 多项式模型：y = 2*x1**2 + 3*x2 + 0.5*x3**3 + 4*x4 - 1.5*x5 + 2.7*x6
    return -2 * x1**5 + 3*x2 + 0.5*x3**3 + 4*x4 - 1.5*x5 + 2.7*x6

def perturb_inputs(input_values, perturbation, index):
    perturbed_inputs = input_values.copy()
    perturbed_inputs[index] += perturbation * input_values[index]
    return perturbed_inputs

def perform_sensitivity_analysis_combined():
    base_inputs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    perturbation_range = np.linspace(-0.05, 0.05, 100)

    # 自定义每个输入变量的颜色
    colors = ['#7FFFD4', '#D3F8E2', '#A1D6E2', '#4FC3F7', '#E4C1F9', '#FFA07A']

    # 自定义每个输入变量的线型
    linestyles = ['solid', '--', '-.', ':', '-', 'dashdot'] # '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'

    # 创建一个新的图像
    fig_combined, ax_combined = plt.subplots(figsize=(10, 6))

    # 循环每个输入变量
    for input_index in range(6):
        base_output = polynomial_model(*base_inputs)
        output_values = []

        # 循环每个扰动
        for perturbation in perturbation_range:
            perturbed_inputs = perturb_inputs(base_inputs, perturbation, input_index)
            perturbed_output = polynomial_model(*perturbed_inputs)
            delta_output = perturbed_output - base_output
            output_values.append(delta_output)

        # 使用自定义颜色和线型绘制每个输入变量的曲线
        plt.plot(perturbation_range, output_values, label=f'x{input_index + 1}', color=colors[input_index], linestyle=linestyles[input_index])

    plt.xlabel('Percentage Change of Input Variables (Δx)')
    plt.ylabel('Change in Model Output (Δy)')
    plt.title('Sensitivity Analysis for Input Variables (Combined)')
    plt.legend(loc='upper left', frameon=False, bbox_to_anchor=(1.0, 1.0))  # 显示图例
    plt.grid(False)  # 去掉背景线条

    # 去掉右边和上边的边框线
    ax_combined.spines['top'].set_visible(False)
    ax_combined.spines['right'].set_visible(False)

    # 保存图形为SVG文件
    plt.savefig('output_plot_combined.svg', format='svg', bbox_inches='tight')

    # 显示图形
    plt.show()

def perform_sensitivity_analysis_individual():
    base_inputs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    perturbation_range = np.linspace(-0.05, 0.05, 100)

    # 自定义每个输入变量的颜色
    colors = ['#7FFFD4', '#D3F8E2', '#A1D6E2', '#4FC3F7', '#E4C1F9', '#FFA07A']

    # 自定义每个输入变量的线型
    linestyles = ['solid', '--', '-.', ':', '-', 'dashdot'] # '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'

    # 循环每个输入变量
    for input_index in range(6):
        # 创建一个新的图像
        fig_individual, ax_individual = plt.subplots(figsize=(6, 4))

        base_output = polynomial_model(*base_inputs)
        output_values = []

        # 循环每个扰动
        for perturbation in perturbation_range:
            perturbed_inputs = perturb_inputs(base_inputs, perturbation, input_index)
            perturbed_output = polynomial_model(*perturbed_inputs)
            delta_output = perturbed_output - base_output
            output_values.append(delta_output)

        # 使用自定义颜色和线型绘制每个输入变量的曲线
        plt.plot(perturbation_range, output_values, label=f'x{input_index + 1}', color=colors[input_index], linestyle=linestyles[input_index])  # 自定义颜色

        plt.xlabel('Percentage Change of Input Variable (Δx)')
        plt.ylabel('Change in Model Output (Δy)')
        plt.title(f'Sensitivity Analysis for x{input_index + 1}')
        plt.grid(False)

        # 去掉右边和上边的边框线
        ax_individual.spines['top'].set_visible(False)
        ax_individual.spines['right'].set_visible(False)

        # 保存图形为SVG文件
        plt.savefig(f'output_plot_individual_x{input_index + 1}.svg', format='svg', bbox_inches='tight')

        # 显示图形
        plt.show()

# 执行一次敏感性分析，绘制六条曲线在同一个图中并保存
perform_sensitivity_analysis_combined()

# 执行六次敏感性分析，绘制单条曲线并保存
perform_sensitivity_analysis_individual()
