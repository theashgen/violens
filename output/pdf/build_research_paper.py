from pathlib import Path
import re, html, math, json
from reportlab.pdfgen import canvas
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, BalancedColumns, KeepTogether, Flowable)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon, Circle
from reportlab.graphics import renderSVG

ROOT=Path(__file__).resolve().parent
TEXT=ROOT/'violence-detection-research-paper.md'
PDF=ROOT/'violence-detection-research-paper.pdf'
REVIEW=json.loads((ROOT/'review-results.json').read_text())
MC_RUN=REVIEW['models']['mc3']['runs'][1]
MONITOR=REVIEW['monitor']['runs']
EXT=REVIEW.get('external_benchmarks',{})
UCF_CONT=REVIEW.get('ucf_continuous',{})
UCF_MC=REVIEW.get('ucf_multicam',{}).get('runs',[])
FIG=ROOT/'figures'; FIG.mkdir(exist_ok=True)
W,H=A4
M=42
GAP=18
CW=(W-2*M-GAP)/2
INK=colors.HexColor('#17324a'); BLUE=colors.HexColor('#315d79'); LIGHT=colors.HexColor('#edf2f5'); RUST=colors.HexColor('#95704d')
styles={
 'body':ParagraphStyle('body',fontName='Times-Roman',fontSize=10.8,leading=13.2,alignment=TA_JUSTIFY,spaceAfter=5.0,splitLongWords=True),
 'h1':ParagraphStyle('h1',fontName='Times-Bold',fontSize=11.2,leading=13.2,alignment=TA_CENTER,spaceBefore=5,spaceAfter=7,keepWithNext=True),
 'h2':ParagraphStyle('h2',fontName='Times-Italic',fontSize=10.4,leading=12.5,spaceBefore=4.5,spaceAfter=4.5,keepWithNext=True),
 'caption':ParagraphStyle('caption',fontName='Times-Roman',fontSize=8.6,leading=10.2,alignment=TA_LEFT,spaceAfter=7),
 'small':ParagraphStyle('small',fontName='Times-Roman',fontSize=8.8,leading=10.4,spaceAfter=4),
 'cell':ParagraphStyle('cell',fontName='Times-Roman',fontSize=8.4,leading=10.2),
 'ref':ParagraphStyle('ref',fontName='Times-Roman',fontSize=8.2,leading=9.8,spaceAfter=4.8,splitLongWords=True),
}
def p(text,sty='body'):
 text=html.escape(text)
 text=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',text)
 text=re.sub(r'`([^`]+)`',r'<font name="Courier" size="8.6">\1</font>',text)
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
 for i,line in enumerate(lines):label(d,x+w/2,y+h/2+(len(lines)-1)*5-i*10-3,line,8.5)
def architecture():
 d=Drawing(CW,210)
 box(d,10,168,CW-20,32,'Offline: audit / split / train\nCompatible MC3-18 checkpoint')
 box(d,10,115,100,32,'Uploaded video\nUniform 16 frames')
 box(d,CW-110,115,100,32,'Camera capture\nRecent time window')
 arrow(d,CW/2,168,CW/2,95)
 box(d,30,61,CW-60,34,'Shared spatial preprocessing\nShared model / serialized inference')
 arrow(d,60,115,70,95);arrow(d,CW-60,115,CW-70,95)
 box(d,10,6,100,32,'Upload response\nClass + confidence')
 box(d,CW-110,6,100,32,'Camera hysteresis\nState / active event')
 arrow(d,70,61,60,38);arrow(d,CW-70,61,CW-60,38)
 return d

def states():
 d=Drawing(CW,170)
 box(d,5,133,99,25,'CONNECTING')
 box(d,CW-104,133,99,25,'BUFFERING')
 arrow(d,104,145,CW-104,145)
 label(d,CW/2,151,'frames arrive',7)
 box(d,5,77,99,25,'NORMAL')
 box(d,CW-104,77,99,25,'VIOLENCE')
 arrow(d,CW-55,133,CW-55,112);arrow(d,CW-55,112,55,112);arrow(d,55,112,55,102)
 label(d,CW/2,117,'first scored window',7)
 arrow(d,104,94,CW-104,94); label(d,CW/2,102,'N_pos >= T_pos',6.8)
 arrow(d,CW-104,83,104,83);label(d,CW/2,67,'N_neg <= T_neg',6.8)
 box(d,48,6,CW-96,25,'OFFLINE / reconnect')
 arrow(d,55,77,55,41);arrow(d,55,41,CW/2,31)
 arrow(d,CW-55,77,CW-55,41);arrow(d,CW-55,41,CW/2,31)
 label(d,CW/2,46,'capture failure clears state',7)
 return d

def confusion():
 d=Drawing(CW,190)
 x0=70;y0=22;s=62
 label(d,x0+s,178,'Predicted class',9,bold=True)
 label(d,x0+s/2,162,'Non-violence',8);label(d,x0+1.5*s,162,'Violence',8)
 for r,row in enumerate(MC_RUN["test_default"]["confusion"]):
  for c,v in enumerate(row):
   x=x0+c*s;y=y0+(1-r)*s
   fill=colors.HexColor('#c9dce7') if r==c else colors.HexColor('#f2e2d8')
   d.add(Rect(x,y,s,s,fillColor=fill,strokeColor=colors.white,strokeWidth=2))
   label(d,x+s/2,y+s/2-4,str(v),15,True)
 label(d,x0-6,y0+1.5*s-3,'Non-violence',8,anchor='end')
 label(d,x0-6,y0+.5*s-3,'Violence',8,anchor='end')
 label(d,9,86,'Actual',8,anchor='start');label(d,9,76,'class',8,anchor='start')
 label(d,x0+s,7,'Rows: actual | columns: predicted',8)
 return d

def metrics():
 d = Drawing(CW, 152)
 left, bottom, right, top = 32, 22, CW - 10, 120
 series = []
 for name, line_color in [('mc3', BLUE), ('resnet', RUST)]:
  runs = REVIEW['models'][name]['runs']
  for key in ('train_loss', 'val_loss'):
   values = [sum(run['history'][i][key] for run in runs)/len(runs) for i in range(20)]
   series.append((key, line_color, values))
 max_loss = math.ceil(max(max(vals) for _, _, vals in series)*5)/5
 for i in range(5):
  y = bottom+(top-bottom)*i/4
  d.add(Line(left,y,right,y,strokeColor=colors.HexColor('#dddfe2'),strokeWidth=.45))
  label(d,left-4,y-2,f'{max_loss*i/4:.2f}',7,anchor='end')
 for epoch in (1,5,10,15,20):
  x = left+(right-left)*(epoch-1)/19
  label(d,x,bottom-9,str(epoch),7)
 for key,line_color,vals in series:
  for i,(a,b) in enumerate(zip(vals, vals[1:])):
   x1=left+(right-left)*i/19
   x2=left+(right-left)*(i+1)/19
   style = {'strokeDashArray': [2, 2]} if key == 'val_loss' else {}
   d.add(Line(x1,bottom+a/max_loss*(top-bottom),x2,bottom+b/max_loss*(top-bottom),
              strokeColor=line_color,strokeWidth=1.1,**style))
 label(d,left,128,'Cross-entropy',8,anchor='start')
 label(d,(left+right)/2,2,'Epoch',8)
 label(d,left,142,'MC3-18',8,True,BLUE,anchor='start')
 label(d,left+45,142,'ResNet-18',8,True,RUST,anchor='start')
 label(d,right,142,'Solid: train  Dashed: validation',7,anchor='end')
 return d

def scaling():
 d = Drawing(CW, 154)
 left, bottom, right, top = 30, 22, CW - 30, 118
 max_tp = 12.0
 max_drop = 32.0
 cams = [r['cameras'] for r in UCF_MC]
 tps = [r['throughput_per_second'] for r in UCF_MC]
 drops = [r['replaced_rate'] * 100.0 for r in UCF_MC]
 for i in range(5):
  y = bottom + (top - bottom) * i / 4
  d.add(Line(left, y, right, y, strokeColor=colors.HexColor('#dddfe2'), strokeWidth=.45))
  label(d, left - 4, y - 2, f'{max_tp*i/4:.0f}', 7, color=BLUE, anchor='end')
  label(d, right + 4, y - 2, f'{max_drop*i/4:.0f}%', 7, color=RUST, anchor='start')
 def x_of(idx):
  return left + (right - left) * idx / max(1, len(cams) - 1)
 for idx, c in enumerate(cams):
  label(d, x_of(idx), bottom - 9, str(c), 7)
 for i in range(len(cams) - 1):
  x1, x2 = x_of(i), x_of(i + 1)
  y1 = bottom + tps[i] / max_tp * (top - bottom)
  y2 = bottom + tps[i + 1] / max_tp * (top - bottom)
  d.add(Line(x1, y1, x2, y2, strokeColor=BLUE, strokeWidth=1.2))
  d1 = bottom + drops[i] / max_drop * (top - bottom)
  d2 = bottom + drops[i + 1] / max_drop * (top - bottom)
  d.add(Line(x1, d1, x2, d2, strokeColor=RUST, strokeWidth=1.1, strokeDashArray=[2, 2]))
 for idx, (tp, dr) in enumerate(zip(tps, drops)):
  d.add(Circle(x_of(idx), bottom + tp / max_tp * (top - bottom), 2.1, fillColor=BLUE, strokeColor=BLUE))
  d.add(Circle(x_of(idx), bottom + dr / max_drop * (top - bottom), 2.1, fillColor=RUST, strokeColor=RUST))
 label(d, left, 126, 'Win/s', 7.5, True, BLUE, anchor='start')
 label(d, right, 126, 'Replaced %', 7.5, True, RUST, anchor='end')
 label(d, (left + right) / 2, 2, 'Concurrent unedited CCTV streams', 7.5)
 label(d, left, 142, 'Throughput (solid)', 7.5, True, BLUE, anchor='start')
 label(d, right, 142, 'Replaced windows (dashed)', 7.5, True, RUST, anchor='end')
 return d

def table(caption,headers,rows,widths=None):
 data=[[p(x,'cell') for x in headers]]+[[p(str(x),'cell') for x in row] for row in rows]
 t=Table(data,colWidths=widths or [CW/len(headers)]*len(headers),hAlign='LEFT')
 t.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),.8,colors.black),('LINEBELOW',(0,0),(-1,0),.5,colors.black),('LINEBELOW',(0,-1),(-1,-1),.8,colors.black),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),3.5),('RIGHTPADDING',(0,0),(-1,-1),3.5),('TOPPADDING',(0,0),(-1,-1),3.2),('BOTTOMPADDING',(0,0),(-1,-1),3.2),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#f3f3f3'))]))
 return KeepTogether([p(caption,'caption'),t,Spacer(1,6)])

figures={
 'FIG_ARCH':(architecture(),'Fig. 1. Implemented data paths. The temporal policies differ; checkpoint and spatial preprocessing are shared.'),
 'FIG_STATES':(states(),'Fig. 2. Monitoring hysteresis state machine. Default parameters are T_pos=0.65 (N_pos=3) and T_neg=0.40 (N_neg=4); validation-calibrated policy sets T_pos=0.85 (N_pos=5).'),
 'FIG_CONFUSION':(confusion(),'Fig. 3. RLVS grouped test confusion matrix for MC3-18 seed 42 at threshold 0.50. Violence is the positive class.'),
 'FIG_METRICS':(metrics(),'Fig. 4. Mean training and validation loss by epoch across three seeds for each backbone on RLVS.'),
 'FIG_SCALING':(scaling(),'Fig. 5. Live SurveillanceManager throughput and replaced-window rate across 1 to 16 simultaneous unedited UCF-Crime CCTV streams (42 s each, MPS).')}
for key,(drawing,_) in figures.items():renderSVG.drawToFile(drawing,str(FIG/(key.lower()+'.svg')))

def special(key):
 if key in figures:
  d,cap=figures[key];return KeepTogether([d,p(cap,'caption')])
 if key == 'TABLE_SPLITS':
  splits = REVIEW['audit']['partitions']
  return table('TABLE I. VISUALLY GROUPED RLVS PARTITIONS', ['Set', 'Non-violence', 'Violence', 'Groups'],
               [[name.title(), *splits[name]['class_counts'], splits[name]['groups']]
                for name in ('train', 'val', 'test')])
 if key == 'TABLE_TRAINING':
  return table('TABLE II. TRAINING AND SELECTION', ['Setting', 'Value'],
               [['Backbones', 'Frozen MC3-18 and ResNet-18'], ['Trainable layer', 'Linear classifier'],
                ['Epochs per seed', 20], ['Batch size / rate', '32 / 0.001'],
                ['Seeds', '41, 42, 43'], ['Optimizer', 'AdamW'],
                ['Epoch selection', 'Lowest validation loss']])
 if key == 'TABLE_CONTINUOUS':
  rows = []
  for label_txt, k in [
   ('Val (0.65 / 3w)', 'val_default'),
   ('Val (0.85 / 5w)', 'val_selected'),
   ('Test (0.65 / 3w)', 'test_default'),
   ('Test (0.85 / 5w)', 'test_calibrated'),
  ]:
   b = UCF_CONT[k]
   rows.append([
    label_txt,
    f"{b['events_matched']}/{b['events']} ({b['event_recall']*100:.1f}%)",
    f"{b['delay_median_sec']:.1f}/{b['delay_mean_sec']:.1f}" if b['delay_median_sec'] is not None else 'N/A',
    f"{b['false_alerts']} ({b['false_alerts_per_camera_hour']:.1f})",
    f"{b['event_f1']*100:.1f}%",
   ])
  return table(
   'TABLE III. UNEDITED UCF-CRIME CONTINUOUS EVENT EVALUATION',
   ['Split (policy)', 'Matched (Rec)', 'Delay (s)', 'FA (/cam-h)', 'Ev. F1'],
   rows,
   [CW*.26, CW*.23, CW*.18, CW*.20, CW*.13],
  )
 if key == 'TABLE_MONITOR':
  return table(
   'TABLE IV. LIVE MULTI-CAMERA SCALING (UNEDITED UCF-CRIME)',
   ['Cams', 'Done / sched.', 'Win/s', 'p50 / p95 ms', 'Replaced'],
   [[
    r['cameras'],
    f"{r['completed']} / {r['scheduled']}",
    f"{r['throughput_per_second']:.2f}",
    f"{r['latency_p50_ms']:.1f} / {r['latency_p95_ms']:.1f}",
    f"{r['replaced_windows']} ({r['replaced_rate']*100:.1f}%)",
   ] for r in UCF_MC],
   [CW*.12, CW*.23, CW*.15, CW*.26, CW*.24],
  )
 if key == 'TABLE_RESULTS':
  summary = [REVIEW['models'][name]['summary']['test_default'] for name in ('mc3', 'resnet')]
  def result_cell(values):
   return f"{100*values['mean']:.2f} +/- {100*values['sd']:.2f}"
  return table('TABLE V. RLVS GROUPED TEST MEAN +/- SD (%) AT 0.50', ['Metric', 'MC3-18', 'ResNet-18'],
               [[metric.title(),*(result_cell(s[metric]) for s in summary)]
                for metric in ('accuracy', 'precision', 'recall', 'f1')])
 if key == 'TABLE_EXTERNAL':
  rows = []
  for label_txt, ext_key in [
   ('RWF-2000 val (400)', 'rwf2000_val'),
   ('Bus Violence (1,400)', 'bus_violence'),
   ('AIRTLab (350)', 'airtlab'),
  ]:
   d = EXT[ext_key]
   for arch_lbl, arch_k in (('MC3', 'mc3'), ('Res18', 'resnet')):
    s = d['models'][arch_k]['summary']['clean_default']
    rows.append([
     f"{label_txt} [{arch_lbl}]",
     f"{100*s['accuracy']['mean']:.1f}+/-{100*s['accuracy']['sd']:.1f}",
     f"{100*s['f1']['mean']:.1f}+/-{100*s['f1']['sd']:.1f}",
     f"{s['roc_auc']['mean'] + 1e-9:.3f}",
    ])
  return table(
   'TABLE VI. ZERO-SHOT EXTERNAL BENCHMARKS (0 RLVS OVERLAP, @0.50)',
   ['Benchmark [Model]', 'Acc (%)', 'F1 (%)', 'AUC'],
   rows,
   [CW*.44, CW*.21, CW*.21, CW*.14],
  )
 if key == 'TABLE_THRESHOLDS':
  chosen = MC_RUN['validation_selected']['threshold']
  default_val = next(r for r in MC_RUN['validation_sweep'] if r['threshold'] == .50)
  rows = [('Val', .5, default_val), ('Val', chosen, MC_RUN['validation_selected']),
          ('Test', .5, MC_RUN['test_default']), ('Test', chosen, MC_RUN['test_selected'])]
  return table('TABLE VII. RLVS VALIDATION-SELECTED CLIP RULE (SEED 42)', ['Set', 'Cutoff', 'Precision', 'Recall', 'F1'],
               [[split, f'{cutoff:.2f}', *(f'{row[k]:.2%}' for k in ('precision','recall','f1'))]
                for split,cutoff,row in rows])
 if key == 'TABLE_LOOKALIKES':
  nv = EXT['airtlab']['models']['mc3']['runs'][1]['nonviolent_action_fp']
  order = ['friendly punch', 'jump', 'hug', 'highfive', 'walk', 'greet', 'handgestures', 'handshake']
  rows = []
  for act in order:
   if act in nv:
    st = nv[act]
    rows.append([act, f"{st['fp_050']} / {st['clips']}", f"{st['fpr_050']*100:.1f}%", f"{st['mean_score']:.2f}"])
  return table(
   'TABLE VIII. AIRTLAB NON-VIOLENT LOOKALIKE FP RATES (MC3 SEED 42)',
   ['Action tag', 'FP / Clips', 'FPR @0.50', 'Mean score'],
   rows,
   [CW*.33, CW*.22, CW*.23, CW*.22],
  )
 if key == 'TABLE_PROVENANCE':
  return table('TABLE IX. EXPERIMENT ARTIFACTS',['Artifact','Contents'],
               [['grouped-splits.csv','RLVS paths, classes, split, group IDs'],
                ['experiments.json','6 runs: loss, val sweeps, RLVS test'],
                ['ext-*-results.json','RWF-2000, Bus Violence, AIRTLab eval'],
                ['ucf-continuous-results.json','UCF-Crime val calibration & test eval'],
                ['ucf-multicam-results.json','1-16 stream SurveillanceManager scaling'],
                ['review-labels.json','AI-assisted 5-frame RLVS clip screen'],
                ['review-results.json','Compact verified data for this paper']],[CW*.46,CW*.54])
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
c.setSubject('RLVS, external benchmark, and continuous multi-camera violence detection evaluation')

# University cover, following the user-supplied centered dissertation layout.
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
 registration = f"Reg. No. {author['registration_number']}"
 roll = f"Roll No. {author['roll_number']}"
 for value, y, font, size in [
  (author['name'], 437, 'Times-Bold', 14),
  (registration, 418, 'Times-Roman', 11),
  (roll, 403, 'Times-Roman', 11),
  (author['email'], 388, 'Times-Roman', 10),
 ]:
  c.setFont(font, size)
  c.drawCentredString(x, y, value)
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
