import wx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_wxagg as wxagg
from datetime import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# Janela de ajuda


class HelpWindow(wx.Frame):
    def __init__(self, *args, **kw):
        super(HelpWindow, self).__init__(*args, **kw)

        self.SetTitle("Ajuda - Manual de Instruções")
        self.SetSize((600, 400))

        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        instructions = """\
Bem-vindo ao Manual de Instruções!

Nome do Programa: Interface Gráfica Básica para Testes com IA
Versão: 1.0.0

Criadores: Adson Borges e Victor Fidelis;
Disciplina Tópicos Especiais em IA (2024.2)
Professor: José Alfredo
UFRN

1. Selecione o modelo desejado no menu suspenso "Selecione Modelo".
2. Insira os parâmetros específicos do modelo no campo apropriado.
3. Selecione o arquivo do dataset clicando no botão "Browse" e escolhendo um arquivo CSV válido.
4. Ajuste a proporção de treino e teste usando o slider "Train-Test Split (Tamanho do teste %)".
5. Clique no botão "Run" para treinar e avaliar o modelo.
6. Os resultados serão exibidos na área de saída e um histórico de execução será mantido abaixo, 
   a direita você poderá ver alguns plots.

Nota:
- A área de parâmetros muda dependendo do modelo selecionado e aceita parâmetros específicos para cada modelo.
- O arquivo CSV deve ter como delimitador a ';' (ponto e vígula).
- O campo "Iterações Máx" é usado apenas para o MLPClassifier e não é aplicável para outros modelos.
- Nem todos os modelos contam com plots específicos, como o MLPClassifier e DecisionTreeClassifier. Mas todos 
  possuem matriz de confusão.

Modelos Disponíveis:
- MLPClassifier: Redes Neurais Multicamadas
- DecisionTreeClassifier: Árvores de Decisão
- RandomForestClassifier: Florestas Aleatórias
- SVC: Máquina de Vetores de Suporte
- KNeighborsClassifier: K-Nearest Neighbors
- LogisticRegression: Regressão Logística
- GaussianNB: Naive Bayes Gaussiano
- GradientBoostingClassifier: Classificador de Gradiente Boosting

Fique a vontade para utilizar o código e inserir melhorias
"""
        instructions_text = wx.TextCtrl(
            panel, value=instructions, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH)

        vbox.Add(instructions_text, proportion=1,
                 flag=wx.EXPAND | wx.ALL, border=10)
        panel.SetSizer(vbox)


class NeuralNetworkGUI(wx.Frame):
    def __init__(self, *args, **kw):
        super(NeuralNetworkGUI, self).__init__(*args, **kw)

        self.SetTitle("Teste com IA")
        self.SetSize((1350, 800))

        panel = wx.Panel(self)
        grid_layout = wx.GridBagSizer(vgap=15, hgap=15)

        # Seletor de Modelo
        hbox0 = wx.BoxSizer(wx.HORIZONTAL)
        self.model_label = wx.StaticText(panel, label="Selecione Modelo:")
        self.model_choice = wx.ComboBox(panel, choices=["MLPClassifier", "DecisionTreeClassifier", "RandomForestClassifier", "SVC",
                                        "KNeighborsClassifier", "LogisticRegression", "GaussianNB", "GradientBoostingClassifier"], style=wx.CB_READONLY)
        self.model_choice.SetSelection(0)
        self.model_choice.Bind(wx.EVT_COMBOBOX, self.on_model_change)

        # Matplotlib figure and axes for animation
        self.fig, self.ax = plt.subplot_mosaic(
            [['left', 'center', 'right'], ['bottom', 'bottom', 'bottom'], ['bottom', 'bottom', 'bottom']], figsize=(8, 7))
        self.fig.tight_layout()
        self.canvas = wxagg.FigureCanvasWxAgg(panel, -1, self.fig)

        hbox0.Add(self.model_label, flag=wx.RIGHT, border=10)
        hbox0.Add(self.model_choice, proportion=1)

        # Input de Hiperparâmetro
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        self.param_label = wx.StaticText(
            panel, label="Camadas ocultas e neurônios (separadas por vírgula):")
        self.param_text = wx.TextCtrl(panel, value="50, 50")

        hbox1.Add(self.param_label, flag=wx.RIGHT, border=60)
        hbox1.Add(self.param_text, proportion=1)

        # Função de Ativação Scikit
        hbox1_1 = wx.BoxSizer(wx.HORIZONTAL)
        self.activation_label = wx.StaticText(
            panel, label="Função de Ativação")
        self.activation_choice = wx.ComboBox(
            panel, choices=["identity", "relu", "tanh", "logistic"], style=wx.CB_READONLY)
        self.activation_choice.SetSelection(0)
        # self.activation_choice.Bind(wx.EVT_COMBOBOX, self.on_model_change)
        hbox1_1.Add(self.activation_label, flag=wx.RIGHT, border=40)
        hbox1_1.Add(self.activation_choice, proportion=1)

        # Seleção do Solver
        hbox1_2 = wx.BoxSizer(wx.HORIZONTAL)
        self.solver_label = wx.StaticText(panel, label="Otimização:")
        self.solver_choice = wx.ComboBox(
            panel, choices=["adam", "sgd", "lbfgs"], style=wx.CB_READONLY)
        self.solver_choice.SetSelection(0)
        # self.solver_choice.Bind(wx.EVT_COMBOBOX, self.on_model_change)
        hbox1_2.Add(self.solver_label, flag=wx.RIGHT, border=40)
        hbox1_2.Add(self.solver_choice, proportion=1)

        # Batch Size
        hbox1_3 = wx.BoxSizer(wx.HORIZONTAL)
        self.batch_size_label = wx.StaticText(panel, label="Tamanho do batch:")
        self.batch_size_text = wx.TextCtrl(panel, value="auto")
        hbox1_3.Add(self.batch_size_label, flag=wx.RIGHT, border=40)
        hbox1_3.Add(self.batch_size_text, proportion=1)

        # Learning Rate
        hbox1_4 = wx.BoxSizer(wx.HORIZONTAL)
        self.learning_rate_label = wx.StaticText(
            panel, label="Taxa de aprendizagem:")
        self.learning_rate_choice = wx.ComboBox(
            panel, choices=["constant", "invscaling", "adaptative"], style=wx.CB_READONLY)
        self.learning_rate_choice.SetSelection(0)
        # self.learning_rate_choice.Bind(wx.EVT_COMBOBOX, self.on_model_change)
        hbox1_4.Add(self.learning_rate_label, flag=wx.RIGHT, border=40)
        hbox1_4.Add(self.learning_rate_choice, proportion=1)

        # Input Max iterations
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        self.max_iter_label = wx.StaticText(panel, label="Iterações (máx):")
        self.max_iter_text = wx.TextCtrl(panel, value="500")
        hbox2.Add(self.max_iter_label, flag=wx.RIGHT, border=40)
        hbox2.Add(self.max_iter_text, proportion=1)

        # Seletor de Arquivo
        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        self.file_label = wx.StaticText(panel, label="Arquivo dataset:")
        self.file_picker = wx.FilePickerCtrl(
            panel, message="Selecione um arquivo")
        hbox3.Add(self.file_label, flag=wx.RIGHT, border=20)
        hbox3.Add(self.file_picker, proportion=1)

        # Slider Train-test split
        hbox4 = wx.BoxSizer(wx.HORIZONTAL)
        self.split_label = wx.StaticText(
            panel, label="Train-Test Split (tamanho do teste %):")
        self.split_slider = wx.Slider(
            panel, value=50, minValue=10, maxValue=90, style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        self.split_slider.Bind(wx.EVT_SLIDER, self.on_slider_change)
        self.split_value = wx.StaticText(panel, label="50%")

        hbox4.Add(self.split_label, flag=wx.RIGHT, border=20)
        hbox4.Add(self.split_slider, proportion=1)
        hbox4.Add(self.split_value, flag=wx.LEFT, border=20)

        # Botão Run
        self.run_button = wx.Button(panel, label="Run")
        self.run_button.Bind(wx.EVT_BUTTON, self.on_run)

        # Botão de Ajuda
        self.help_button = wx.Button(panel, label="Ajuda")
        self.help_button.Bind(wx.EVT_BUTTON, self.on_help)

        # Botão de Salvar Histórico
        self.save_button = wx.Button(panel, label="Salvar Histórico")
        self.save_button.Bind(wx.EVT_BUTTON, self.on_save)

        # Output Area
        self.output_area = wx.TextCtrl(
            panel, style=wx.TE_MULTILINE | wx.TE_READONLY)

        # Area do Historico
        self.history_area = wx.TextCtrl(
            panel, style=wx.TE_MULTILINE | wx.TE_READONLY)
        self.history_label = wx.StaticText(
            panel, label="Histórico de Performance:")

        # Add sizers
        grid_layout.Add(hbox0, pos=(0, 0), flag=wx.EXPAND |
                        wx.LEFT | wx.TOP, border=10)
        grid_layout.Add(hbox1, pos=(1, 0), flag=wx.EXPAND | wx.LEFT, border=10)
        grid_layout.Add(hbox1_1, pos=(2, 0),
                        flag=wx.EXPAND | wx.LEFT, border=10)
        grid_layout.Add(hbox1_2, pos=(3, 0),
                        flag=wx.EXPAND | wx.LEFT, border=10)
        grid_layout.Add(hbox1_3, pos=(4, 0),
                        flag=wx.EXPAND | wx.LEFT, border=10)
        grid_layout.Add(hbox1_4, pos=(5, 0),
                        flag=wx.EXPAND | wx.LEFT, border=10)
        grid_layout.Add(hbox2, pos=(6, 0), flag=wx.EXPAND | wx.LEFT, border=10)
        grid_layout.Add(hbox3, pos=(7, 0), flag=wx.EXPAND | wx.LEFT, border=10)
        grid_layout.Add(hbox4, pos=(8, 0), flag=wx.EXPAND | wx.LEFT, border=10)
        grid_layout.Add(self.run_button, pos=(9, 0),
                        flag=wx.ALIGN_LEFT | wx.LEFT, border=10)
        grid_layout.Add(self.output_area, pos=(
            10, 0), span=(1, 2), flag=wx.EXPAND | wx.LEFT, border=10)
        grid_layout.Add(self.history_label, pos=(11, 0),
                        flag=wx.ALIGN_LEFT | wx.LEFT, border=10)
        grid_layout.Add(self.history_area, pos=(12, 0), span=(
            2, 2), flag=wx.EXPAND | wx.LEFT, border=10)
        grid_layout.Add(self.save_button, pos=(14, 1), flag=wx.RIGHT)
        grid_layout.Add(self.help_button, pos=(16, 8),
                        flag=wx.ALIGN_RIGHT | wx.RIGHT | wx.BOTTOM, border=5)
        grid_layout.Add(self.canvas, pos=(0, 2), span=(15, 8),
                        flag=wx.LEFT | wx.LEFT | wx.TOP, border=10)

        grid_layout.AddGrowableCol(0, proportion=1)
        grid_layout.AddGrowableCol(1, proportion=1)
        grid_layout.AddGrowableCol(2, proportion=1)
        grid_layout.AddGrowableCol(3, proportion=1)
        grid_layout.AddGrowableCol(4, proportion=1)
        grid_layout.AddGrowableCol(5, proportion=1)
        grid_layout.AddGrowableCol(6, proportion=1)
        grid_layout.AddGrowableCol(7, proportion=1)
        grid_layout.AddGrowableCol(8, proportion=1)
        grid_layout.AddGrowableCol(9, proportion=1)

        grid_layout.AddGrowableRow(0, proportion=1)
        grid_layout.AddGrowableRow(1, proportion=1)
        grid_layout.AddGrowableRow(2, proportion=1)
        grid_layout.AddGrowableRow(3, proportion=1)
        grid_layout.AddGrowableRow(4, proportion=1)
        grid_layout.AddGrowableRow(5, proportion=1)
        grid_layout.AddGrowableRow(6, proportion=1)
        grid_layout.AddGrowableRow(7, proportion=1)
        grid_layout.AddGrowableRow(8, proportion=1)
        grid_layout.AddGrowableRow(9, proportion=1)
        grid_layout.AddGrowableRow(10, proportion=1)
        grid_layout.AddGrowableRow(11, proportion=1)
        grid_layout.AddGrowableRow(12, proportion=1)
        grid_layout.AddGrowableRow(13, proportion=1)
        grid_layout.AddGrowableRow(14, proportion=1)

        panel.SetSizer(grid_layout)

        # Inicializar history list
        self.history = []

    def on_model_change(self, event):
        model_name = self.model_choice.GetValue()

        if model_name in ["MLPClassifier"]:
            self.param_label.SetLabel(
                "Camadas ocultas e neurônios (separadas por vírgula):")
            self.param_text.SetValue("50,50")
            self.param_text.Enable()
            self.max_iter_label.Enable()
            self.max_iter_text.Enable()
            self.activation_label.Enable()
            self.activation_choice.Enable()
            self.solver_label.Enable()
            self.solver_choice.Enable()
            self.batch_size_label.Enable()
            self.batch_size_text.Enable()
            self.learning_rate_label.Enable()
            self.learning_rate_choice.Enable()
        elif model_name == "DecisionTreeClassifier":
            self.param_label.SetLabel("Max Depth:")
            self.param_text.SetValue("None")
            self.param_text.Enable()
            self.max_iter_label.Disable()
            self.max_iter_text.Disable()
            self.activation_label.Disable()
            self.activation_choice.Disable()
            self.solver_label.Disable()
            self.solver_choice.Disable()
            self.batch_size_label.Disable()
            self.batch_size_text.Disable()
            self.learning_rate_label.Disable()
            self.learning_rate_choice.Disable()
        elif model_name == "RandomForestClassifier":
            self.param_label.SetLabel("Número de Estimadores:")
            self.param_text.SetValue("100")
            self.param_text.Enable()
            self.max_iter_label.Disable()
            self.max_iter_text.Disable()
            self.activation_label.Disable()
            self.activation_choice.Disable()
            self.solver_label.Disable()
            self.solver_choice.Disable()
            self.batch_size_label.Disable()
            self.batch_size_text.Disable()
            self.learning_rate_label.Disable()
            self.learning_rate_choice.Disable()
        elif model_name == "SVC":
            self.param_label.SetLabel("Kernel:")
            self.param_text.SetValue("linear")
            self.param_text.Enable()
            self.max_iter_label.Disable()
            self.max_iter_text.Disable()
            self.activation_label.Disable()
            self.activation_choice.Disable()
            self.solver_label.Disable()
            self.solver_choice.Disable()
            self.batch_size_label.Disable()
            self.batch_size_text.Disable()
            self.learning_rate_label.Disable()
            self.learning_rate_choice.Disable()
        elif model_name == "KNeighborsClassifier":
            self.param_label.SetLabel("Número de vizinhos:")
            self.param_text.SetValue("5")
            self.param_text.Enable()
            self.max_iter_label.Disable()
            self.max_iter_text.Disable()
            self.activation_label.Disable()
            self.activation_choice.Disable()
            self.solver_label.Disable()
            self.solver_choice.Disable()
            self.batch_size_label.Disable()
            self.batch_size_text.Disable()
            self.learning_rate_label.Disable()
            self.learning_rate_choice.Disable()
        elif model_name == "LogisticRegression":
            self.param_label.SetLabel("Inverse Regularization Strength (C):")
            self.param_text.SetValue("1.0")
            self.param_text.Enable()
            self.max_iter_label.Disable()
            self.max_iter_text.Disable()
            self.activation_label.Disable()
            self.activation_choice.Disable()
            self.solver_label.Disable()
            self.solver_choice.Disable()
            self.batch_size_label.Disable()
            self.batch_size_text.Disable()
            self.learning_rate_label.Disable()
            self.learning_rate_choice.Disable()
        elif model_name == "GaussianNB":
            self.param_label.SetLabel("Parametro")
            self.param_text.Disable()
            self.max_iter_label.Disable()
            self.max_iter_text.Disable()
            self.activation_label.Disable()
            self.activation_choice.Disable()
            self.solver_label.Disable()
            self.solver_choice.Disable()
            self.batch_size_label.Disable()
            self.batch_size_text.Disable()
            self.learning_rate_label.Disable()
            self.learning_rate_choice.Disable()
        elif model_name == "GradientBoostingClassifier":
            self.param_label.SetLabel("Número de Estimadores:")
            self.param_text.SetValue("100")
            self.param_text.Enable()
            self.max_iter_label.Disable()
            self.max_iter_text.Disable()
            self.activation_label.Disable()
            self.activation_choice.Disable()
            self.solver_label.Disable()
            self.solver_choice.Disable()
            self.batch_size_label.Disable()
            self.batch_size_text.Disable()
            self.learning_rate_label.Disable()
            self.learning_rate_choice.Disable()

        self.Layout()

    def on_slider_change(self, event):
        self.split_value.SetLabel(f"{self.split_slider.GetValue()}%")

    def on_run(self, event):
        try:
            # Carregar e processar dados
            file_path = self.file_picker.GetPath()
            if not os.path.isfile(file_path):
                wx.MessageBox(
                    "Caminho Inválido. Por favor, insira um arquivo válido.", "Error", wx.ICON_ERROR)
                return

            # Carregar dado do arquivo e manejar erros
            try:
                data = pd.read_csv(file_path, encoding='latin1',
                                   delimiter=';', on_bad_lines='skip')
            except Exception as e:
                wx.MessageBox(f"Erro ao ler arquivo: {
                              str(e)}", "Error", wx.ICON_ERROR)
                return

            if data.shape[1] < 2:
                wx.MessageBox(
                    "Dataset deve ter ao menos uma coluna feature e uma coluna target.", "Error", wx.ICON_ERROR)
                return

            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values

            if X.shape[1] == 0:
                wx.MessageBox("Features não encontrados no dataset.",
                              "Error", wx.ICON_ERROR)
                return

            test_size = self.split_slider.GetValue() / 100
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42)
            self.X_train, self.X_test = self.standardize_data(
                self.X_train, self.X_test)

            # Captura de parâmetros e inicialização do modelo
            model_name = self.model_choice.GetValue()
            activation_function = self.activation_choice.GetValue()
            solver_algorithm = self.solver_choice.GetValue()
            learning_rate = self.learning_rate_choice.GetValue()

            if self.batch_size_text.GetValue() != "auto":
                batch_size = int(self.batch_size_text.GetValue())
            else:
                batch_size = self.batch_size_text.GetValue()

            if model_name == "MLPClassifier":
                self.ax['left'].clear()
                self.ax['center'].clear()
                self.ax['bottom'].clear()
                hidden_layer_sizes = tuple(
                    map(int, self.param_text.GetValue().split(',')))
                max_iter = int(self.max_iter_text.GetValue())
                model = MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, activation=activation_function, solver=solver_algorithm, batch_size=batch_size, learning_rate=learning_rate, random_state=42, verbose=True)

            elif model_name == "DecisionTreeClassifier":
                self.ax['left'].clear()
                self.ax['center'].clear()
                self.ax['bottom'].clear()
                max_depth = None if self.param_text.GetValue(
                ) == "None" else int(self.param_text.GetValue())
                model = DecisionTreeClassifier(
                    max_depth=max_depth, random_state=42)

            elif model_name == "RandomForestClassifier":
                self.ax['left'].clear()
                self.ax['center'].clear()
                self.ax['bottom'].clear()
                n_estimators = int(self.param_text.GetValue())
                model = RandomForestClassifier(
                    n_estimators=n_estimators, random_state=42)

            elif model_name == "SVC":
                self.ax['left'].clear()
                self.ax['center'].clear()
                self.ax['bottom'].clear()
                kernel = self.param_text.GetValue()
                model = SVC(kernel=kernel, random_state=42)

            elif model_name == "KNeighborsClassifier":
                self.ax['left'].clear()
                self.ax['center'].clear()
                self.ax['bottom'].clear()
                n_neighbors = int(self.param_text.GetValue())
                model = KNeighborsClassifier(n_neighbors=n_neighbors)

            elif model_name == "LogisticRegression":
                self.ax['left'].clear()
                self.ax['center'].clear()
                self.ax['bottom'].clear()
                C = float(self.param_text.GetValue())
                model = LogisticRegression(C=C, random_state=42)

            elif model_name == "GaussianNB":
                self.ax['left'].clear()
                self.ax['center'].clear()
                self.ax['bottom'].clear()
                model = GaussianNB()

            elif model_name == "GradientBoostingClassifier":
                self.ax['left'].clear()
                self.ax['center'].clear()
                self.ax['bottom'].clear()
                n_estimators = int(self.param_text.GetValue())
                model = GradientBoostingClassifier(
                    n_estimators=n_estimators, random_state=42)

            # Treinar modelo
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(
                self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1_scr = f1_score(self.y_test, y_pred, average='weighted')

            # Atualiza histórico e output
            if model_name == "MLPClassifier":
                result_text = f"Modelo: {model_name}; Camadas: {self.param_text.GetValue()}; Teste%:{test_size}; Max_Iter: {max_iter}; Actv: {activation_function}; Solver: {solver_algorithm} \nAcuracia: {
                    accuracy:.4f}; Precisão: {precision:.4f}; Recall: {recall:.4f}; F1_score:{f1_scr:.4f}"
                self.output_area.SetValue(result_text)
                self.history.append(datetime.now().strftime("%D %H:%M:%S"))
                self.history.append(result_text)
                self.history_area.SetValue("\n\n".join(self.history))
            else:
                result_text = f"Modelo: {model_name}, Teste%:{test_size}; Parâmetro: {self.param_text.GetValue()} \nAcuracia: {
                    accuracy:.4f}; Precisão: {precision:.4f}; Recall: {recall:.4f}; F1_score:{f1_scr:.4f}"
                self.output_area.SetValue(result_text)
                self.history.append(datetime.now().strftime("%D %H:%M:%S"))
                self.history.append(result_text)
                self.history.append("\n")
                self.history_area.SetValue("\n".join(self.history))

            # Plot estrutura da rede e progresso do treinamento for MLPClassifier
            if model_name == "MLPClassifier":
                if self.solver_choice.GetValue() != "lbfgs":
                    losses = model.loss_curve_
                    self.plot_training_progress(losses)
                else:
                    self.ax['center'].clear()
                self.plot_network_structure(hidden_layer_sizes)

            elif model_name == "DecisionTreeClassifier":
                self.ax['left'].clear()
                self.ax['center'].clear()
                self.ax['bottom'].clear()
                plot_tree(model, filled=True, ax=self.ax['bottom'])

            elif model_name == "RandomForestClassifier":
                self.ax['left'].clear()
                self.ax['center'].clear()
                self.ax['bottom'].clear()
                plot_tree(model.estimators_[
                          0], filled=True, ax=self.ax['bottom'])

            # Plot Matriz de confusão
            self.plot_confusion_matrix(self.y_test, y_pred)

        except Exception as e:
            wx.MessageBox(f"Um erro ocorreu: {
                          str(e)}", "Error", wx.ICON_ERROR)

    def standardize_data(self, X_train, X_test):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test

    def plot_network_structure(self, hidden_layer_sizes):
        self.ax['left'].clear()
        self.ax['left'].set_title("Estrutura da Rede")

        # Número de camadas (input + hidden + output)
        layers = len(hidden_layer_sizes) + 2
        neurons = [self.X_train.shape[1]] + \
            list(hidden_layer_sizes) + [len(np.unique(self.y_train))]

        # Plotar estrutura da rede
        for i in range(layers):
            self.ax['left'].plot([i] * neurons[i],
                                 np.arange(neurons[i]), 'o', label=f'Layer {i+1}')
        self.ax['left'].set_xticks(range(layers))
        self.ax['left'].set_xticklabels(
            [f'Layer {i+1}' for i in range(layers)])
        self.ax['left'].set_yticks(np.arange(0, max(neurons) + 5, 5))
        self.ax['left'].set_ylim(0, max(neurons) + 5)
        self.ax['left'].legend(loc='upper left')
        self.ax['left'].grid(True)

        self.canvas.draw()

    def plot_training_progress(self, losses):
        self.ax['center'].clear()
        self.ax['center'].set_title("Progresso do Treino")
        self.ax['center'].plot(losses, label='Loss')
        self.ax['center'].set_xlabel('Epoch')
        self.ax['center'].set_ylabel('Loss')
        self.ax['center'].legend(loc='upper right')
        self.ax['center'].grid(True)

        self.canvas.draw()

    def plot_confusion_matrix(self, y_test, y_pred):
        self.fig.delaxes(self.ax['right'])
        self.ax['right'] = self.fig.add_subplot(3, 3, 3)

        self.ax['right'].set_title("Matriz de Confusão")
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=self.ax['right'], cmap=plt.cm.Blues, colorbar=False)

        self.canvas.draw()

    def on_help(self, event):
        help_window = HelpWindow(None, title="Ajuda - Manual de Instruções")
        help_window.Show()

    def on_save(self, event):
        dlg = wx.FileDialog(self, "Salvar histórico como...", "", "",
                            "Text files (*.txt)|*.txt", wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dlg.ShowModal() == wx.ID_CANCEL:
            return

        path = dlg.GetPath()
        with open(path, 'w') as file:
            file.write("\n\n".join(self.history))
        dlg.Destroy()


if __name__ == '__main__':
    app = wx.App(False)
    frame = NeuralNetworkGUI(None)
    frame.Show()
    app.MainLoop()
