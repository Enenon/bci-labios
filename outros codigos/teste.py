import wx
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import \
    NavigationToolbar2WxAgg as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np

class MyFrame(wx.Frame):
    def __init__(self,parent,title):
        super(MyFrame, self).__init__(parent, title=title)
        self.panel = wx.Panel(self)
        self.sizer = wx.GridBagSizer(4,3)
        txtTestando = 'carregando'
        self.texto = wx.StaticText(self.panel,label=txtTestando)

        self.timer = wx.Timer(self)
        
        

        self.sizer.Add(self.texto,pos=(0,0))
        self.Bind(wx.EVT_TIMER,self.on_timer,self.timer)

        botao1 = wx.Button(self.panel,label='clique aqui')
        self.sizer.Add(botao1,pos=(1,0))
        botao1.Bind(wx.EVT_BUTTON,self.clicado)
        self.timer.Start(1000)

        self.panel.SetSizer(self.sizer)

        #self.canvas.draw() # redesenha o canvas
        #self.Fit()

    def on_timer(self,event):
        texto = self.texto.GetLabel()
        
        if len(texto) > 15:
            texto = 'carregando'
        else: texto += '.'
        self.texto.SetLabel(texto)

        
    def clicado(self,event):
        Frame2(None,title='oioi').Show()

class Frame2(wx.Frame):
    def __init__(self,parent,title):
        super(Frame2,self).__init__(parent,title=title)
        self.panel=wx.Panel(self)
        self.sizer = wx.BoxSizer()
        self.figure = Figure()
        self.canvas = FigureCanvas(self,-1,self.figure)
        self.sizer.Add(self.canvas)
        self.axes = self.figure.add_subplot()
        self.axes.plot([1,6,9,8,2,4],[4,5,6,8,4,7])
        self.axes.set
        self.Fit()

class MyApp(wx.App):
    def OnInit(self):
        self.frame = MyFrame(parent=None, title='Teste')
        self.frame.Show()

        return True
    
app = MyApp()
app.MainLoop()