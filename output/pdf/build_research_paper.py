from pathlib import Path
import re, html, math, json
from reportlab.pdfgen import canvas
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, BalancedColumns, KeepTogether, Flowable)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.graphics import renderSVG

ROOT=Path(__file__).resolve().parent
TEXT=ROOT/'violence-detection-research-paper.md'
PDF=ROOT/'violence-detection-research-paper.pdf'
FIG=ROOT/'figures'; FIG.mkdir(exist_ok=True)
W,H=A4
M=42
GAP=18
CW=(W-2*M-GAP)/2
INK=colors.HexColor('#17324a'); BLUE=colors.HexColor('#315d79'); LIGHT=colors.HexColor('#edf2f5')
styles={
 'body':ParagraphStyle('body',fontName='Times-Roman',fontSize=10.5,leading=12.7,alignment=TA_JUSTIFY,spaceAfter=5.5,splitLongWords=True),
 'h1':ParagraphStyle('h1',fontName='Times-Bold',fontSize=11.5,leading=13.5,alignment=TA_CENTER,spaceBefore=6,spaceAfter=8,keepWithNext=True),
 'h2':ParagraphStyle('h2',fontName='Times-Italic',fontSize=10.7,leading=13,spaceBefore=5,spaceAfter=5,keepWithNext=True),
 'caption':ParagraphStyle('caption',fontName='Times-Roman',fontSize=9,leading=10.5,alignment=TA_LEFT,spaceAfter=8),
 'small':ParagraphStyle('small',fontName='Times-Roman',fontSize=9,leading=10.5,spaceAfter=4),
 'cell':ParagraphStyle('cell',fontName='Times-Roman',fontSize=9,leading=11),
 'ref':ParagraphStyle('ref',fontName='Times-Roman',fontSize=8.5,leading=10.3,spaceAfter=6,splitLongWords=True),
}
def p(text,sty='body'):
 text=html.escape(text)
 text=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',text)
 text=re.sub(r'(https?://[^\s]+)',lambda m:f'<link href="{m.group(0).rstrip(chr(46))}" color="#17324a">{m.group(0)}</link>',text)
 return Paragraph(text,styles[sty])
def label(d,x,y,text,size=9,bold=False,color=colors.black,anchor='middle'):
 d.add(String(x,y,text,fontName='Times-Bold' if bold else 'Times-Roman',fontSize=size,fillColor=color,textAnchor=anchor))
def arrow(d,x1,y1,x2,y2,color=BLUE):
 d.add(Line(x1,y1,x2,y2,strokeColor=color,strokeWidth=.8))
 a=math.atan2(y2-y1,x2-x1);s=4
 d.add(Polygon([x2,y2,x2-s*math.cos(a-.5),y2-s*math.sin(a-.5),x2-s*math.cos(a+.5),y2-s*math.sin(a+.5)],fillColor=color,strokeColor=color))
def box(d,x,y,w,h,text):
 d.add(Rect(x,y,w,h,rx=3,ry=3,strokeColor=BLUE,fillColor=LIGHT,strokeWidth=.7))
 lines=text.split('\n')
 for i,line in enumerate(lines):label(d,x+w/2,y+h/2+(len(lines)-1)*5-i*10-3,line,8.7)
def architecture():
 d=Drawing(CW,215)
 box(d,10,173,CW-20,32,'Offline: audit / split / train\nCompatible MC3-18 checkpoint')
 box(d,10,119,100,32,'Uploaded video\nUniform 16 frames')
 box(d,CW-110,119,100,32,'Camera capture\nRecent time window')
 arrow(d,CW/2,173,CW/2,98)
 box(d,30,64,CW-60,34,'Shared spatial preprocessing\nShared model / serialized inference')
 arrow(d,60,119,70,98);arrow(d,CW-60,119,CW-70,98)
 box(d,10,8,100,32,'Upload response\nClass + confidence')
 box(d,CW-110,8,100,32,'Camera hysteresis\nState / active event')
 arrow(d,70,64,60,40);arrow(d,CW-70,64,CW-60,40)
 return d

def states():
 d=Drawing(CW,175)
 box(d,5,137,99,25,'CONNECTING')
 box(d,CW-104,137,99,25,'BUFFERING')
 arrow(d,104,149,CW-104,149)
 label(d,CW/2,155,'frames arrive',7)
 box(d,5,80,99,25,'NORMAL')
 box(d,CW-104,80,99,25,'VIOLENCE')
 arrow(d,CW-55,137,CW-55,116);arrow(d,CW-55,116,55,116);arrow(d,55,116,55,105)
 label(d,CW/2,121,'first scored window',7)
 arrow(d,104,97,CW-104,97); label(d,CW/2,105,'3 scores >= 0.65',7)
 arrow(d,CW-104,86,104,86);label(d,CW/2,72,'4 scores <= 0.40',7)
 box(d,48,7,CW-96,25,'OFFLINE / reconnect')
 arrow(d,55,80,55,43);arrow(d,55,43,CW/2,32)
 arrow(d,CW-55,80,CW-55,43);arrow(d,CW-55,43,CW/2,32)
 label(d,CW/2,48,'capture failure clears state',7)
 return d

def confusion():
 d=Drawing(CW,200)
 x0=70;y0=25;s=65
 label(d,x0+s,188,'Predicted class',9,bold=True)
 label(d,x0+s/2,171,'Non-violence',8);label(d,x0+1.5*s,171,'Violence',8)
 for r,row in enumerate([[102,16],[1,119]]):
  for c,v in enumerate(row):
   x=x0+c*s;y=y0+(1-r)*s
   fill=colors.HexColor('#c9dce7') if r==c else colors.HexColor('#f2e2d8')
   d.add(Rect(x,y,s,s,fillColor=fill,strokeColor=colors.white,strokeWidth=2))
   label(d,x+s/2,y+s/2-4,str(v),16,True)
 label(d,x0-6,y0+1.5*s-3,'Non-violence',8,anchor='end')
 label(d,x0-6,y0+.5*s-3,'Violence',8,anchor='end')
 label(d,9,91,'Actual',8,anchor='start');label(d,9,81,'class',8,anchor='start')
 label(d,x0+s,9,'Rows: actual | columns: predicted',8)
 return d

def metrics():
 d=Drawing(CW,185)
 vals=[('Accuracy',89.1688,92.8571),('Precision',82.7004,88.1481),('Recall',98.9899,99.1667),('F1',90.1149,93.3333)]
 x0=49; maxw=CW-82
 for tick in [0,25,50,75,100]:
  x=x0+maxw*tick/100
  d.add(Line(x,33,x,161,strokeColor=colors.HexColor('#dddddd'),strokeWidth=.4))
  label(d,x,20,str(tick),7)
 for i,(name,a,b) in enumerate(vals):
  y=145-i*30
  label(d,x0-5,y-1,name,8,anchor='end')
  for offset,val,color in [(4,a,colors.HexColor('#b3c0ca')),(-7,b,BLUE)]:
   d.add(Rect(x0,y+offset,maxw*val/100,7,strokeColor=None,fillColor=color))
   label(d,x0+maxw*val/100+3,y+offset+.5,f'{val:.1f}',6.8,anchor='start')
 d.add(Rect(48,174,8,7,strokeColor=None,fillColor=colors.HexColor('#b3c0ca')));label(d,60,174,'Validation (n=397)',8,anchor='start')
 d.add(Rect(151,174,8,7,strokeColor=None,fillColor=BLUE));label(d,163,174,'Test (n=238)',8,anchor='start')
 label(d,x0+maxw/2,6,'Percentage (%)',8)
 return d

def table(caption,headers,rows,widths=None):
 data=[[p(x,'cell') for x in headers]]+[[p(str(x),'cell') for x in row] for row in rows]
 t=Table(data,colWidths=widths or [CW/len(headers)]*len(headers),hAlign='LEFT')
 t.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),.8,colors.black),('LINEBELOW',(0,0),(-1,0),.5,colors.black),('LINEBELOW',(0,-1),(-1,-1),.8,colors.black),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#f3f3f3'))]))
 return KeepTogether([p(caption,'caption'),t,Spacer(1,8)])

figures={
 'FIG_ARCH':(architecture(),'Fig. 1. Implemented data paths. The temporal policies differ; checkpoint and spatial preprocessing are shared.'),
 'FIG_STATES':(states(),'Fig. 2. Simplified monitoring state logic. Threshold counts are defaults. Mid-range scores reset counters; inference failures use an ERROR state (not shown).'),
 'FIG_CONFUSION':(confusion(),'Fig. 3. Test confusion matrix from the saved 238-clip evaluation. Violence is the positive class.'),
 'FIG_METRICS':(metrics(),'Fig. 4. Recorded validation and test metrics for the same one-epoch checkpoint. The partitions differ; these bars are not a longitudinal improvement curve.')}
for key,(drawing,_) in figures.items():renderSVG.drawToFile(drawing,str(FIG/(key.lower()+'.svg')))

def special(key):
 if key in figures:
  d,cap=figures[key];return KeepTogether([d,p(cap,'caption')])
 if key=='TABLE_SPLITS':return table('TABLE I. FINAL RETAINED VIDEO PARTITIONS',['Split','Non-violence','Violence','Total'],[['Train',674,672,1346],['Validation',199,198,397],['Test',118,120,238],['Total',991,990,1981]],[CW*.24,CW*.27,CW*.24,CW*.25])
 if key=='TABLE_TRAINING':return table('TABLE II. RECORDED CHECKPOINT CONFIGURATION',['Setting','Recorded value'],[['Backbone','MC3-18 / Kinetics-400'],['Trainable layers','Linear classifier only'],['Completed epochs','1'],['Batch size / learning rate','4 / 0.001'],['Seed / frame count','42 / 16'],['Optimizer','AdamW'],['Selection criterion','Minimum validation loss'],['Training / validation clips','1,346 / 397']],[CW*.44,CW*.56])
 if key=='TABLE_API':return table('TABLE III. UPLOADED-VIDEO RESPONSE CONTRACT',['Status','Meaning'],[['200','Valid prediction returned'],['413','Upload exceeds size limit'],['415','Unsupported filename extension'],['422','Empty, corrupt, or over-limit video'],['429','Shared inference model busy'],['503','Trained checkpoint unavailable'],['500','Model execution failure']],[CW*.18,CW*.82])
 if key=='TABLE_RESULTS':return table('TABLE IV. PRIMARY HELD-OUT TEST RESULTS',['Metric','Value'],[['Test clips','238'],['Accuracy','92.86%'],['Precision (violence)','88.15%'],['Recall (violence)','99.17%'],['F1 (violence)','93.33%'],['Specificity (derived)','86.44%'],['Balanced accuracy (derived)','92.80%'],['Cross-entropy loss','0.2092']],[CW*.68,CW*.32])
 if key=='TABLE_PROVENANCE':return table('TABLE V. LOCAL EXPERIMENT EVIDENCE',['Artifact','Purpose'],[['rlvs-inspection.json','Decoded records and audit failures'],['rlvs-exclusions.csv','Five reviewed exclusions'],['splits.csv','1,981 retained assignments'],['best.pt','Epoch-1 model and metadata'],['training.json','One recorded epoch'],['test_metrics.json','238-clip primary evaluation']],[CW*.46,CW*.54])
 raise ValueError(key)

def flows(text):
 result=[]; in_refs=False
 for block in re.split(r'\n\s*\n',text.strip()):
  block=' '.join(block.splitlines()).strip()
  if not block:continue
  if block.startswith('## '):
   title=block[3:];in_refs=title=='References';result.append(p(title,'h1'))
  elif block.startswith('### '):result.append(p(block[4:],'h2'))
  elif re.fullmatch(r'\[\[\w+\]\]',block):result.append(special(block[2:-2]))
  elif in_refs:result.append(p(block,'ref'))
  elif block.startswith('**Keywords:'):result.append(p(block,'small'))
  else:result.append(p(block))
 return result

class Group(Flowable):
 def __init__(self,items):
  super().__init__();self.items=items
 def wrap(self,w,h):
  self.w=w; self.heights=[]
  for f in self.items:
   a,b=f.wrapOn(self.canv,w,100000)
   self.heights.append(b+f.getSpaceBefore()+f.getSpaceAfter())
  self.height=sum(self.heights);self.width=w
  return w,self.height
 def draw(self):
  y=self.height
  for f,h in zip(self.items,self.heights):
   y-=f.getSpaceBefore();_,hh=f.wrapOn(self.canv,self.w,100000)
   f.drawOn(self.canv,0,y-hh);y-=hh+f.getSpaceAfter()

def normalize(fs):
 return [Group(f._content) if isinstance(f,KeepTogether) else f for f in fs]

def pack(items,colheight,c,draw=False,top=0):
 queue=list(items); col=0;used=0;useds=[]
 while queue:
  f=queue.pop(0)
  _,h=f.wrapOn(c,CW,100000)
  before=f.getSpaceBefore() if used else 0
  after=f.getSpaceAfter()
  available=colheight-used-before
  # Move subsection and section headings with at least the beginning of their text.
  reserve=0
  if f.getKeepWithNext() and queue:
   reserve=min(40,queue[0].wrapOn(c,CW,100000)[1])
  if h+reserve>available+.01:
   pieces=f.splitOn(c,CW,available) if isinstance(f,Paragraph) and not reserve and available>40 else []
   if pieces:
    first=pieces[0];_,hh=first.wrapOn(c,CW,available)
    if draw:first.drawOn(c,M+col*(CW+GAP),top-used-before-hh)
    used+=before+hh
    queue=pieces[1:]+queue
   else:queue.insert(0,f)
   useds.append(used);col+=1;used=0
   if col==2:return False,useds,queue
   continue
  if draw:f.drawOn(c,M+col*(CW+GAP),top-used-before-h)
  used+=before+h+after
 useds.append(used)
 return True,useds,[]

text=TEXT.read_text();sections=text.split('<!-- page -->')
title=sections[0].split('\n\n')[0][2:]
sections[0]='## Abstract'+sections[0].split('## Abstract',1)[1]
c=canvas.Canvas(str(PDF),pagesize=A4)
cover=json.loads((ROOT/'cover-details.json').read_text())
c.setTitle(title);c.setAuthor('; '.join(a['name'] for a in cover['authors']))
c.setSubject('RLVS video classification and continuous multi-camera research prototype')

# University cover, following the user-supplied centered dissertation layout.
cover=json.loads((ROOT/'cover-details.json').read_text())
def centered(text,y,font='Times-Roman',size=14):
 c.setFillColor(colors.black);c.setFont(font,size);c.drawCentredString(W/2,y,text)
cover_title=Paragraph(
 'Violence Detection in Surveillance Videos<br/>Using Transfer Learning and<br/>Continuous Multi-Camera Monitoring',
 ParagraphStyle('coverTitle',fontName='Times-Bold',fontSize=24,leading=29,alignment=TA_CENTER))
_,title_height=cover_title.wrap(W-100,180)
cover_title.drawOn(c,50,H-92-title_height)
centered('RESEARCH PROJECT REPORT',607,'Times-Bold',14)
centered('Submitted in partial fulfilment of the',586,'Times-Italic',12)
centered('requirements for the award of the degree of',571,'Times-Italic',12)
centered(cover['degree'],535,'Times-Bold',14)
centered('IN',516,'Times-Bold',14)
centered(cover['specialization'],497,'Times-Bold',14)
centered('By',461,'Times-Italic',12)
for i,author in enumerate(cover['authors']):
 x=110+i*(W-220)/2
 for value,y,font,size in [(author['name'],437,'Times-Bold',14),('Reg. No. '+author['registration_number'],418,'Times-Roman',11),(author['email'],401,'Times-Roman',10)]:
  c.setFont(font,size);c.drawCentredString(x,y,value)
if cover['supervisor']:
 centered('Under the guidance of',363,'Times-Italic',12)
 centered(cover['supervisor'],338,'Times-Bold',14)
logo_width=230;logo_height=logo_width*810/1600
c.drawImage(str(FIG/'lpu-logo.jpg'),(W-logo_width)/2,193,width=logo_width,height=logo_height,mask='auto')
centered(cover['school'],163,'Times-Bold',14)
centered(cover['university'],144,'Times-Bold',14)
centered(cover['location'],125,'Times-Bold',14)
centered(cover['submission_date'],99,'Times-Italic',12)
centered('i',23,'Times-Roman',9)
c.showPage()
for idx,section in enumerate(sections):
 top=H-42
 if idx==0:
  ts=ParagraphStyle('title',fontName='Times-Bold',fontSize=18,leading=21.5,alignment=TA_CENTER)
  ps=Paragraph(title,ts);_,ht=ps.wrap(W-2*M,150);ps.drawOn(c,M,top-ht);top-=ht+13
  for i,author in enumerate(cover['authors']):
   x=110+i*(W-220)/2
   for value,offset,font,size in [(author['name'],0,'Times-Bold',11),('Reg. No. '+author['registration_number'],13,'Times-Roman',9),(author['email'],25,'Times-Roman',9)]:
    c.setFont(font,size);c.drawCentredString(x,top-offset,value)
  top-=43
  c.setFont('Times-Roman',10);c.drawCentredString(W/2,top,'B.Tech in Artificial Intelligence and Machine Learning');top-=13
  c.drawCentredString(W/2,top,'Lovely Professional University, Punjab, India');top-=24
 items=normalize(flows(section))
 maxh=top-44
 ok,_,rem=pack(items,maxh,c)
 if not ok:raise RuntimeError(f'Page {idx+1} overflow: {len(rem)} blocks remain')
 lo,hi=100,maxh
 for _ in range(20):
  mid=(lo+hi)/2
  if pack(items,mid,c)[0]:hi=mid
  else:lo=mid
 ok,used,_=pack(items,hi+.1,c,draw=True,top=top)
 print('Page',idx+1,'available',round(maxh),'used',[round(x) for x in used])
 c.setFont('Times-Roman',9);c.drawCentredString(W/2,23,str(idx+1))
 c.showPage()
c.save()
print(PDF)
