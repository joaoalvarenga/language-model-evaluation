import unittest
from asr_language_model_evaluation.preprocessing import normalize


class TestPreprocess(unittest.TestCase):

    def test_url(self):
        text = 'No Buraco do Aipo, com muitas pedras, ela aparece concentrada. http://blogoosfero.cc/ilhadomel/pousadasilhadomel.com.br/espuma-marrom-chamada-ninguem-merece'
        expected_text = 'no buraco do aipo com muitas pedras ela aparece concentrada'
        self.assertEqual(normalize(text), expected_text)

    def test_html_tag(self):
        text = '<p>Reconhecimento de fala é uma área interdisciplinar originária da <a href="/wiki/Lingu%C3%ADstica_computacional" title="Linguística computacional">linguística computacional</a> cujo objetivo é desenvolver métodos e tecnologias que permitam o reconhecimento e a transcrição de linguagem falada de maneira automática. As tecnologias de reconhecimento de fala são normalmente conhecidas pela sigla em inglês <b>ASR</b> de <b>Automatic Speech Recognition</b> (reconhecimento automático de fala), <b>Computer Speech Recognition</b> (reconhecimento de fala por computador) ou <b>STT</b> de <b>Speech to Text</b> (fala para texto).</p>'
        expected_text = 'reconhecimento de fala é uma área interdisciplinar originária da linguística computacional cujo objetivo é desenvolver métodos e tecnologias que permitam o reconhecimento e a transcrição de linguagem falada de maneira automática as tecnologias de reconhecimento de fala são normalmente conhecidas pela sigla em inglês asr de automatic speech recognition reconhecimento automático de fala computer speech recognition reconhecimento de fala por computador ou stt de speech to text fala para texto'
        self.assertEqual(normalize(text), expected_text)

    def test_percentage(self):
        text = 'hoje o índice cresceu 12% acima do normal'
        expected_text = 'hoje o índice cresceu doze porcento acima do normal'
        self.assertEqual(normalize(text), expected_text)

    def test_filled_pause(self):
        text = 'eu nem ã gosto disso muito éh não sei como vai hum ser'
        expected_text = 'eu nem ah gosto disso muito eh não sei como vai uh ser'
        self.assertEqual(normalize(text), expected_text)

        text = 'ã eu nem gosto disso muito éh não sei como vai hum ser ã'
        expected_text = 'ah eu nem gosto disso muito eh não sei como vai uh ser ah'
        self.assertEqual(normalize(text), expected_text)

        text = 'éh eu nem gosto disso muito éh não sei como vai hum ser éh'
        expected_text = 'eh eu nem gosto disso muito eh não sei como vai uh ser eh'
        self.assertEqual(normalize(text), expected_text)

        text = 'hum eu nem gosto disso muito éh não sei como vai hum ser hum'
        expected_text = 'uh eu nem gosto disso muito eh não sei como vai uh ser uh'
        self.assertEqual(normalize(text), expected_text)

    def test_numbers(self):
        text = 'hoje 30 de abril de 1529, eu ganhei 1000 reais'
        expected_text = 'hoje trinta de abril de mil quinhentos e vinte e nove eu ganhei mil reais'
        self.assertEqual(normalize(text), expected_text)

    def test_money(self):
        text = 'nossa ganhei R$ 5000 só nessa jogada'
        expected_text = 'nossa ganhei cinco mil reais só nessa jogada'
        self.assertEqual(normalize(text), expected_text)

        text = 'essa bala custou R$ 0,50'
        expected_text = 'essa bala custou cinquenta centavos'
        self.assertEqual(normalize(text), expected_text)

        text = 'essa bala custou R$ 0,0'
        expected_text = 'essa bala custou zero reais'
        self.assertEqual(normalize(text), expected_text)

        text = 'muito caro esse salgado a R$2,50'
        expected_text = 'muito caro esse salgado a dois reais e cinquenta centavos'
        self.assertEqual(normalize(text), expected_text)

        text = 'R$1.400,00'
        expected_text = 'mil e quatrocentos reais'
        self.assertEqual(normalize(text), expected_text)

        text = 'R$15.421,12'
        expected_text = 'quinze mil quatrocentos e vinte e um reais e doze centavos'
        self.assertEqual(normalize(text), expected_text)

        text = 'R$1,01'
        expected_text = 'um real e um centavo'
        self.assertEqual(normalize(text), expected_text)

        text = 'R$350.000'
        expected_text = 'trezentos e cinquenta mil reais'
        self.assertEqual(normalize(text), expected_text)

    def test_hours(self):
        text = 'Que horas começa o evento mesmo? Às 15h30'
        expected_text = 'que horas começa o evento mesmo às quinze horas e trinta minutos'
        self.assertEqual(normalize(text), expected_text)

        text = 'Que horas começa o evento mesmo? Às 15:30'
        expected_text = 'que horas começa o evento mesmo às quinze horas e trinta minutos'
        self.assertEqual(normalize(text), expected_text)

        text = 'O café da manhã começa às 8h da manhã'
        expected_text = 'o café da manhã começa às oito horas da manhã'
        self.assertEqual(normalize(text), expected_text)

    def test_metrics(self):
        text = '1.032m²Superfície'
        expected_text = 'mil e trinta e dois metros quadrados superfície'
        self.assertEqual(normalize(text), expected_text)

        text = '500m da praia.'
        expected_text = 'quinhentos metros da praia'
        self.assertEqual(normalize(text), expected_text)

        text = '2 lotes em excelente local de Toque Toque Grande. mais de 1000m2 para você realizar seu sonho de construir na melhor praia do litoral norte.'
        expected_text = 'dois lotes em excelente local de toque toque grande mais de mil metros quadrados para você realizar seu sonho de construir na melhor praia do litoral norte'
        self.assertEqual(normalize(text), expected_text)

        text = 'Ótima topografia com frente de 40m.'
        expected_text = 'ótima topografia com frente de quarenta metros'
        self.assertEqual(normalize(text), expected_text)

    def test_date(self):
        text = 'Este quarto já está disponível, outro quarto de casal estará dia 10/04.'
        expected_text = 'este quarto já está disponível outro quarto de casal estará dia dez do quatro'
        self.assertEqual(normalize(text), expected_text)

        text = 'Ele nasceu dia 04/08/1996, um dia histórico...'
        expected_text = 'ele nasceu dia quatro do oito de mil novecentos e noventa e seis um dia histórico'
        self.assertEqual(normalize(text), expected_text)

    def test_apostrophe(self):
        text = 'n\'esse mundo a transfigurar-se n\'essas anciãs para novas e mais bellas expressões da vida n\'essa esperança luminosa e feiticeira e apesar do deslumbramento da visão as atribulações do momento venciam no tudo'
        expected_text = 'nesse mundo a transfigurar se nessas anciãs para novas e mais bellas expressões da vida nessa esperança luminosa e feiticeira e apesar do deslumbramento da visão as atribulações do momento venciam no tudo'
        self.assertEqual(normalize(text), expected_text)

        text = 'aqui não se treme não cáe neve porque mamãe você se lembra d\'aquelle chapéo que você tirou do menino na rua e me deu'
        expected_text = 'aqui não se treme não cáe neve porque mamãe você se lembra daquelle chapéo que você tirou do menino na rua e me deu'
        self.assertEqual(normalize(text), expected_text)