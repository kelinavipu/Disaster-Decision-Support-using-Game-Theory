# Install once: pip install dash plotly pandas numpy
# Run: python dss_app.py   (opens http://127.0.0.1:8050 in a new browser tab)
import re, numpy as np, pandas as pd, itertools, math
LV=["Moderate","Severe","Critical"]; # all capacities / priors / weights are USER inputs (see cfg)

def beliefs(rep, r, prior):
    """Bayes: uniform prior, report likelihood r if true==report else (1-r)/2."""
    p=np.array([r if l==rep else (1-r)/2 for l in LV])*np.array(prior,float); return p/p.sum()

def build(c):
    R=c['regions']; P=np.array([f['p'] for f in c['futures']],float)
    post=np.array([beliefs(x['sev'],c['rel'],c['prior']) for x in R])
    return dict(n=len(R),names=[x['name'] for x in R],pop=np.array([float(x['pop']) for x in R]),post=post,
        sevE=post@np.array(c['wts'],float),w=np.array(c['wts'],float),cap=(c['cb'],c['ct'],c['pk']),own=(lambda o:o if o.sum()>0 else np.array([float(x['pop']) for x in R]))(np.array([float(x.get('own') or 0) for x in R])),P=P/P.sum(),m=np.array([f['m'] for f in c['futures']],float),
        S=np.array([c['boats'],c['teams'],c['food']],float),fn=[f['name'] for f in c['futures']])

def util(M,F,s,sev=None):
    """Utility of share F[i] of ALL resources to region i in future s: severity-weighted people x mean coverage over (boats,teams,food)."""
    need=M['pop']*(M['sevE'] if sev is None else sev)
    d=np.stack([need/M['cap'][0],need/M['cap'][1],M['pop']/M['cap'][2]],-1)*M['m'][s]
    return need*np.minimum(1,np.asarray(F)[...,None]*M['S']/d).mean(-1)

def grid(n):
    N=20 if n<=3 else 10
    G=[c for c in itertools.product(range(N+1),repeat=n-1) if sum(c)<=N]
    return np.array([list(c)+[N-sum(c)] for c in G])/N

def Wall(M,F): return np.stack([util(M,F,s).sum(-1) for s in range(len(M['m']))],-1)
def evalf(M,f):
    W=Wall(M,np.asarray(f)[None])[0]; return W,float(W@M['P']),float(W.min())

def sub(M,idx,e): return {**M,'n':len(idx),'pop':M['pop'][idx],'sevE':M['sevE'][idx],'S':M['S']*e}
def shapley(M):
    n=M['n']; tot=M['pop'].sum()
    V={c:(float((Wall(sub(M,list(c),M['own'][list(c)].sum()/M['own'].sum()),grid(len(c)))@M['P']).max()) if c else 0.)
       for k in range(n+1) for c in itertools.combinations(range(n),k)}
    phi=np.zeros(n)
    for perm in itertools.permutations(range(n)):
        cur=()
        for i in perm: nx=tuple(sorted(cur+(i,))); phi[i]+=V[nx]-V[cur]; cur=nx
    return phi/math.factorial(n),V

def solve(c):
    """Core engine: Bayes -> EU / minimax / Shapley -> blended recommendation."""
    M=build(c); F=grid(M['n']); W=Wall(M,F); eu=W@M['P']; wc=W.min(1)
    fe,fm=F[eu.argmax()],F[wc.argmax()]; phi,V=shapley(M)
    shn=phi/phi.sum() if phi.sum()>0 else np.ones(M['n'])/M['n']
    f=(1-c['fair'])*((1-c['risk'])*fe+c['risk']*fm)+c['fair']*shn; f=f/f.sum()
    pp=M['pop']/M['pop'].sum()
    cands={"Expected-utility optimum":fe,"Minimax (worst-case) optimum":fm,"Population-proportional":pp,"Shapley shares":shn,"RECOMMENDED blend":f}
    rows=[]
    for k,v in cands.items():
        w,e,m=evalf(M,v); rows.append({"Strategy":k,**{n:round(x) for n,x in zip(M['fn'],w)},"Expected":round(e),"Worst-case":round(m),
                                       **{M['names'][i]:f"{v[i]:.0%}" for i in range(M['n'])}})
    return dict(M=M,f=f,phi=phi,V=V,fe=fe,fm=fm,shn=shn,nG=len(F),cmp=pd.DataFrame(rows))

def rnd(total,f):
    x=f*total; b=np.floor(x).astype(int)
    for i in np.argsort(-(x-b))[:int(round(total-b.sum()))]: b[i]+=1
    return b

def npay(M,a,greed,cost,s=None):
    a=np.asarray(a); cl=M['pop']*M['sevE']; cl=cl/cl.sum()*(1+greed*a); f=cl/cl.sum()
    u=sum(M['P'][t]*util(M,f,t) for t in range(len(M['P']))) if s is None else util(M,f,s)
    return u*(1-cost*a*((a.sum()-a)>0))      # selfish regions pay a conflict cost if another is selfish too

def nash(M,greed,cost):
    pr=list(itertools.product([0,1],repeat=M['n'])); py={p:npay(M,p,greed,cost) for p in pr}
    eq={p for p in pr if all(py[p][i]>=py[p[:i]+(1-p[i],)+p[i+1:]][i]-1e-9 for i in range(M['n']))}
    return pd.DataFrame([{"Profile (C=cooperate, S=selfish)":" ".join(f"{M['names'][i]}:{'S' if x else 'C'}" for i,x in enumerate(p)),
        **{M['names'][i]:round(py[p][i]) for i in range(M['n'])},"Total":round(py[p].sum()),"Nash equilibrium":"YES" if p in eq else ""} for p in pr])

def stack(M,aud,infl,pen):
    true=M['sevE']; rep=np.minimum(M['w'].max(),true*infl); n=M['n']; rows=[]; best=None
    def pay(lam,a):
        a=np.array(a); w=lam*M['pop']*np.where(a==1,rep,true)+(1-lam)*M['pop']; f=w/w.sum()
        return sum(M['P'][t]*util(M,f,t,true) for t in range(len(M['P'])))*(1-pen*aud*a),f
    for lam in np.linspace(0,1,11):
        pr=list(itertools.product([0,1],repeat=n)); py={p:pay(lam,p)[0] for p in pr}
        eq=[p for p in pr if all(py[p][i]>=py[p[:i]+(1-p[i],)+p[i+1:]][i]-1e-9 for i in range(n))] or [tuple([0]*n)]
        p=max(eq,key=lambda q:py[q].sum()); L=py[p].sum()
        rows.append({"Weight on reports (λ)":round(lam,1),"Followers' equilibrium":" ".join(f"{M['names'][i]}:{'Inflate' if x else 'Honest'}" for i,x in enumerate(p)),"Leader welfare":round(L)})
        if best is None or L>best[0]: best=(L,lam)
    return pd.DataFrame(rows),best[1]

def repeated(M,greed,cost,path):
    n=M['n']; j=int(np.argmax(M['pop']*M['sevE'])); k3=f"{M['names'][j]} defects in round 2, grim punishment"
    cum={"All cooperate":np.zeros(n),"All selfish":np.zeros(n),k3:np.zeros(n)}; out=[]
    for r,s in enumerate(path):
        a3=np.zeros(n,int)
        if r==1: a3[j]=1
        elif r>1: a3[:]=1
        for k,a in zip(cum,[np.zeros(n,int),np.ones(n,int),a3]):
            cum[k]=cum[k]+npay(M,a,greed,cost,s); out.append({"Round":r+1,"Strategy":k,"Cumulative welfare":cum[k].sum(),"Defector's own":cum[k][j]})
    return pd.DataFrame(out),M['names'][j]
from dash import Dash,dcc,html,dash_table,Input,Output,State,ctx,no_update
import plotly.express as px, plotly.graph_objects as go
px.defaults.color_discrete_sequence=['#2f6f6a','#3b6f9a','#5a9a7a','#264b6b','#7fb3a5','#4a7c59']
SEVO=[{'label':l,'value':l} for l in LV]
def tbl(df,**k): return dash_table.DataTable(data=df.to_dict('records'),columns=[{'name':c,'id':c} for c in df.columns],
    style_cell={'padding':'6px','fontFamily':'sans-serif','fontSize':13},style_header={'fontWeight':'bold','backgroundColor':'#2f6f6a','color':'white'},style_table={'overflowX':'auto'},**k)
def num(i,v): return dcc.Input(id=i,type='number',value=v,style={'width':90,'marginRight':16})
def sl(i,v): return dcc.Slider(id=i,min=0,max=1,step=0.05,value=v,marks={0:'0',0.5:'.5',1:'1'},tooltip={'always_visible':False})
PAL=['#2f6f6a','#3b6f9a','#4a7c59','#264b6b','#5a8f9a','#3d6b52']
def box(t,*c): return html.Div([html.H4(t,style={'color':PAL[sum(map(ord,t))%6],'margin':'0 0 8px'}),*c],style={'background':'white','padding':'12px 16px','borderRadius':12,'margin':'12px 0','borderLeft':f"6px solid {PAL[sum(map(ord,t))%6]}",'boxShadow':'0 2px 10px rgba(47,111,106,.14)'})
app=Dash(__name__,suppress_callback_exceptions=True)
app.index_string=app.index_string.replace('</head>','<style>body{background:linear-gradient(135deg,#f1f6f5,#eaf1f6 50%,#eef5f0);margin:0}button{cursor:pointer;border:0;border-radius:8px;padding:8px 14px;background:#2f6f6a;color:white;font-weight:600}input{border:1px solid #a9c5be;border-radius:6px;padding:4px}</style></head>')
ANCH=[(18.989,73.110),(18.910,73.322),(18.737,73.095),(18.785,73.345),(18.437,73.119),(18.083,73.419)]   # land towns: Panvel, Karjat, Pen, Khopoli, Roha, Mahad
_SM=getattr(px,'scatter_map',None)
def mapkw(center,zoom): d=dict(style='open-street-map',center=dict(lat=center[0],lon=center[1]),zoom=zoom); return {'map':d} if _SM else {'mapbox':d}
def base_fig():
    Sm=getattr(go,'Scattermap',None) or go.Scattermapbox
    f=go.Figure(Sm(lat=[18.95],lon=[73.1],mode='markers',marker=dict(size=1,opacity=0),hoverinfo='skip'))
    f.update_layout(height=600,margin=dict(l=0,r=0,t=50,b=0),title=dict(text='Mumbai · Raigad region: press Analyse to overlay the impact heatmap',font=dict(size=14)),**mapkw((18.88,73.22),10)); return f
IDS=['boats','teams','food','cb','ct','pk','pr1','pr2','pr3','w1','w2','w3','rel','risk','fair','greed','cost','aud','infl','pen','path']
EX=dict(boats=10,teams=8,food=500,cb=60,ct=50,pk=1,pr1=.33,pr2=.34,pr3=.33,w1=1,w2=2,w3=3,rel=.7,risk=.3,fair=.2,greed=.5,cost=.2,aud=.5,infl=1.5,pen=.5,path='1,2,3,2')
LBL=dict(boats='Boats available',teams='Rescue teams available',food='Food packets available',cb='People one boat can serve',ct='People one team can serve',pk='Packets needed per person',
 pr1='Prior P(Moderate)',pr2='Prior P(Severe)',pr3='Prior P(Critical)',w1='Severity weight: Moderate',w2='Severity weight: Severe',w3='Severity weight: Critical',
 rel='Report reliability (0-1)',risk='Risk aversion (0=expected utility, 1=minimax)',fair='Fairness weight toward Shapley (0-1)',greed='Greed: extra claim of a selfish region (0-1+)',
 cost='Conflict cost between selfish regions (0-1)',aud='Audit probability (0-1)',infl='Report inflation factor (>=1)',pen='Loss if caught inflating (0-1)',path='Futures per round, 1-based e.g. 1,2,3')
def field(i): return html.Div([html.Label(LBL[i],style={'display':'inline-block','width':330}),dcc.Input(id=i,type='text' if i=='path' else 'number',placeholder='enter value',style={'width':110})],style={'margin':'4px 0'})
RC=['Region','Population','Reported severity','Lat','Lon','Own share %']
FC=['Future','Probability','Severity multiplier']
def blank(cols,n): return [{c:None for c in cols} for _ in range(n)]
REG_EX=[dict(zip(RC,['A',500,'Severe',18.989,73.110,None])),dict(zip(RC,['B',300,'Moderate',18.910,73.322,None])),dict(zip(RC,['C',200,'Critical',18.737,73.095,None]))]
FUT_EX=[dict(zip(FC,x)) for x in[('Stable',.5,1.0),('Worsens',.3,1.3),('Extreme',.2,1.7)]]
app.layout=html.Div(style={'maxWidth':1100,'margin':'auto','fontFamily':'sans-serif'},children=[
 html.Div([html.H2("Game-Theory Disaster Decision Support",style={'margin':0}),html.P("Bayes · Nash · Shapley · Stackelberg · Repeated games · Minimax",style={'margin':'4px 0 0','opacity':.9})],style={'background':'linear-gradient(90deg,#264b6b,#2f6f6a,#4a7c59)','color':'white','padding':'18px 24px','borderRadius':14,'margin':'12px 0'}),
 html.Div([dcc.Tabs(id='tabs',value='p1',colors=dict(border='#d5e4e0',primary='#2f6f6a',background='#e6f0ee'),children=[dcc.Tab(label="01 Configuration",value="p1"),dcc.Tab(label="02 Map",value="p2"),dcc.Tab(label="03 Game theory",value="p3"),dcc.Tab(label="04 Allocation",value="p4"),dcc.Tab(label="05 What-if",value="p5"),dcc.Tab(label="06 How it was processed",value="p6")]),
  html.Div(id='p1',children=[
   html.P("Everything starts empty. Fill every field, or click Load example to see a sample. Nothing is assumed by the code."),
   html.Button("Load example",id='ex'),html.Button("Clear all",id='clr',style={'marginLeft':8}),
   box("1. Regions (Lat/Lon optional: blank = placed automatically on land near Mumbai/Raigad. Own share % optional: the region's own share of resources, used for coalitions; blank = proportional to population)",
     dash_table.DataTable(id='reg',data=blank(RC,3),editable=True,row_deletable=True,columns=[{'name':c,'id':c,'type':'text' if c in('Region','Reported severity') else 'numeric','presentation':'dropdown' if c=='Reported severity' else 'input'} for c in RC],
       dropdown={'Reported severity':{'options':SEVO}}),html.Button("+ Add region (max 6)",id='add',style={'marginTop':8})),
   box("2. Resources",*[field(i) for i in('boats','teams','food')]),
   box("3. Capacity per resource unit",*[field(i) for i in('cb','ct','pk')]),
   box("4. Possible future conditions (probabilities are normalised; multiplier scales demand)",
     dash_table.DataTable(id='fut',data=blank(FC,3),editable=True,row_deletable=True,columns=[{'name':c,'id':c,'type':'text' if c=='Future' else 'numeric'} for c in FC]),html.Button("+ Add future (max 6)",id='addf',style={'marginTop':8})),
   box("5. Uncertainty: prior beliefs, severity weights, report reliability",*[field(i) for i in('pr1','pr2','pr3','w1','w2','w3','rel')]),
   box("6. Strategy settings",*[field(i) for i in('risk','fair','greed','cost','aud','infl','pen','path')])]),
  html.Div(id='p2',children=[html.H4('Interactive flood-region map'),dcc.Graph(id='map',figure=base_fig(),style={'height':620},config={'responsive':True}),html.Div(id='mapinfo',style={'background':'#ecf5f2','border':'1px solid #b7d4cb','borderRadius':8,'padding':'10px 14px','margin':'8px 0'})]),
  html.Div(id='p3',children=[html.Div(id='games')]),
  html.Div(id='p4',children=[html.Div(id='alloc')]),
  html.Div(id='p5',children=[box("Change assumptions vs. the last analysis",
     html.Span("Δ boats "),num('db',0),html.Span("Δ teams "),num('dt',0),html.Span("Δ P(last future), points "),num('dp',0),html.Span("Reliability "),num('r2',None),
     html.Button("Run what-if",id='wi')),html.Div(id='wiout')]),html.Div(id='p6',children=[html.Div(id='trace')])]),
 html.Button("▶ Analyse Disaster Scenario",id='go',style={'fontSize':16,'padding':'10px 20px','margin':'12px 0','background':'linear-gradient(90deg,#264b6b,#2f6f6a)','color':'white','border':0,'borderRadius':6,'position':'sticky','bottom':10,'zIndex':10}),
 html.Div(id='status',style={'color':'#264b6b','fontWeight':'bold'}),dcc.Store(id='cfg')])

@app.callback(Output('reg','data',allow_duplicate=True),Input('add','n_clicks'),State('reg','data'),prevent_initial_call=True)
def add(_,d): return d+blank(RC,1) if len(d)<6 else d
@app.callback(Output('fut','data',allow_duplicate=True),Input('addf','n_clicks'),State('fut','data'),prevent_initial_call=True)
def addf(_,d): return d+blank(FC,1) if len(d)<6 else d
@app.callback([Output(i,'value') for i in IDS]+[Output('reg','data'),Output('fut','data')],Input('ex','n_clicks'),Input('clr','n_clicks'),prevent_initial_call=True)
def load(a,b):
    return [EX[i] for i in IDS]+[REG_EX,FUT_EX] if ctx.triggered_id=='ex' else [None]*len(IDS)+[blank(RC,3),blank(FC,3)]
@app.callback([Output(f'p{i}','style') for i in range(1,7)],Input('tabs','value'))
def show(v): return [({} if v==f'p{i}' else {'position':'absolute','left':'-10000px','width':'1100px','visibility':'hidden'}) for i in range(1,7)]
@app.callback(Output('tabs','value'),Input('cfg','data'),prevent_initial_call=True)
def jump(c): return 'p4' if c else no_update
app.clientside_callback("function(v){setTimeout(function(){window.dispatchEvent(new Event('resize'))},200);return ''}",Output('go','title'),Input('tabs','value'))
def validate(reg,fut,v):
    e=[]
    if len(reg)<2: e.append("Enter at least 2 regions (name, population, severity, lat, lon).")
    for r in reg:
        if not r.get('Region') or not (r.get('Population') or 0)>0 or r.get('Reported severity') not in LV: e.append(f"Incomplete region row: {r.get('Region')}")
    if not fut: e.append("Enter at least 1 future.")
    for x in fut:
        if not x.get('Future') or not (x.get('Probability') or 0)>0 or not (x.get('Severity multiplier') or 0)>0: e.append(f"Incomplete future row: {x.get('Future')}")
    e+=[f"Missing input: {LBL[k]}" for k,x in v.items() if x in (None,'')]
    if not e:
        if sum(v[k] for k in('pr1','pr2','pr3'))<=0: e.append("Priors must sum above 0.")
        e+=[f"{LBL[k]} must be between 0 and 1" for k in('rel','risk','fair','cost','aud','pen') if not 0<=v[k]<=1]
        e+=[f"{LBL[k]} must be > 0" for k in('cb','ct','pk','w1','w2','w3') if v[k]<=0]
        if v['infl']<1: e.append("Inflation factor must be >= 1")
        if not re.fullmatch(r'\s*\d+(\s*,\s*\d+)*\s*',str(v['path'])) or any(not 0<int(t)<=len(fut) for t in str(v['path']).split(',')): e.append("Rounds path must list future numbers between 1 and the number of futures.")
    return e
def mkcfg(reg,fut,v):
    return dict(regions=[dict(name=r['Region'],pop=r['Population'],sev=r['Reported severity'],lat=(r['Lat'] if r.get('Lat') is not None and r.get('Lon') is not None else ANCH[k%6][0]),lon=(r['Lon'] if r.get('Lat') is not None and r.get('Lon') is not None else ANCH[k%6][1]),own=r.get('Own share %')) for k,r in enumerate(reg)],
      futures=[dict(name=x['Future'],p=x['Probability'],m=x['Severity multiplier']) for x in fut],prior=[v['pr1'],v['pr2'],v['pr3']],wts=[v['w1'],v['w2'],v['w3']],**{k:v[k] for k in v if k not in('pr1','pr2','pr3','w1','w2','w3')})
@app.callback(Output('map','figure'),Output('games','children'),Output('alloc','children'),Output('cfg','data'),Output('trace','children'),Output('status','children'),Output('mapinfo','children'),Input('go','n_clicks'),
  State('reg','data'),State('fut','data'),*[State(i,'value') for i in IDS],prevent_initial_call=True)
def run(_,reg,fut,*vals):
    clean=lambda d:[r for r in d if any(x not in (None,'') for x in r.values())]
    reg,fut=clean(reg),clean(fut); v=dict(zip(IDS,vals)); err=validate(reg,fut,v)
    if err:
        msg=box("Please fix these inputs",html.Ul([html.Li(x) for x in err]))
        return no_update,msg,msg,no_update,msg,'Some inputs are missing or invalid: see the list on the Game theory / Allocation tabs, or fix them below.',no_update
    c=mkcfg(reg,fut,v); R=solve(c); M=R['M']; f=R['f']; n=M['n']
    path=[int(x)-1 for x in c['path'].split(',') if x.strip().isdigit() and 0<int(x)<=len(M['m'])] or [0]
    B0,T0,F0=rnd(c['boats'],f),rnd(c['teams'],f),rnd(c['food'],f); pc=M['post'][:,2]; disp=[x if len(x)>2 else f'Region {x}' for x in M['names']]
    auto=sum(1 for r in reg if r.get('Lat') is None or r.get('Lon') is None)
    df=pd.DataFrame(dict(Region=disp,lat=[r['lat'] for r in c['regions']],lon=[r['lon'] for r in c['regions']],Population=M['pop'],P_critical=pc.round(3),
        Share_pct=(f*100).round(1),Boats=B0,Teams=T0,Packets=F0))
    sm=getattr(px,'scatter_map',None); Sm=getattr(go,'Scattermap',None) or go.Scattermapbox
    la,lo=df.lat.mean(),df.lon.mean(); span=max(np.ptp(df.lat),np.ptp(df.lon),0.05); zoom=float(np.clip(8.5-np.log2(span),2,12))
    fig=(sm or px.scatter_mapbox)(df,lat='lat',lon='lon',size='Population',color='P_critical',hover_name='Region',text='Region',size_max=45,opacity=.85,zoom=zoom,center=dict(lat=la,lon=lo),
        range_color=[0,1],color_continuous_scale=['#c3dccb','#8fbf9f','#4f8f6a','#1f5a3f'],hover_data={'lat':False,'lon':False,'P_critical':':.2f','Share_pct':True,'Boats':True,'Teams':True,'Packets':True,'Population':True})
    sevn=(M['sevE']-M['w'].min())/max(M['w'].max()-M['w'].min(),1e-9)
    fig.update_traces(textposition='top right',customdata=np.stack([M['pop'],B0,T0,F0,f,sevn,pc],-1),
        hovertemplate="<b>%{hovertext}</b><br><br>population=%{customdata[0]:.0f}<br>allocated_boats=%{customdata[1]:.0f}<br>allocated_teams=%{customdata[2]:.0f}<br>allocated_packets=%{customdata[3]:.0f}<br>priority_score=%{customdata[4]:.3f}<br>severity=%{customdata[5]:.2f}<br>P(critical)=%{customdata[6]:.2f}<extra></extra>")
    for i in range(n): fig.add_trace(Sm(lat=[la,df.lat[i]],lon=[lo,df.lon[i]],mode='lines',line=dict(width=1+f[i]*10,color='#1f4e5f'),opacity=.6,hoverinfo='skip',showlegend=False))
    Dm=getattr(go,'Densitymap',None) or go.Densitymapbox; rng=np.random.default_rng(0)
    wi=M['pop']*M['sevE']*float(M['P']@M['m']); wi=wi/wi.max(); hl,ho,hz=[],[],[]
    for i in range(n):
        sd=0.012+0.03*np.sqrt(M['pop'][i]/M['pop'].max()); p=rng.normal(0,sd,(120,2)); hl+=list(df.lat[i]+p[:,0]); ho+=list(df.lon[i]+p[:,1]); hz+=[wi[i]]*120
    fig=go.Figure(layout=fig.layout,data=[Dm(lat=hl,lon=ho,z=hz,radius=38,showscale=False,hoverinfo='skip',colorscale=[[0,'rgba(158,197,216,0)'],[.25,'rgba(158,197,216,.45)'],[.55,'rgba(79,143,174,.6)'],[.8,'rgba(44,95,130,.75)'],[1,'rgba(22,56,79,.88)']])]+list(fig.data))
    fig.update_layout(**({'map_style':'open-street-map'} if sm else {'mapbox_style':'open-street-map'}),height=600,margin=dict(l=0,r=0,t=50,b=0),
        title=dict(text='Heat = possible impact intensity; bubbles = population (colour = P(critical)); blue lines = share of resources',font=dict(size=14)),coloraxis_colorbar_title='P(critical)')
    top=int(np.argmax(pc)); info=[html.B(f"Highest posterior probability of critical conditions: {pc[top]:.1%} ({M['names'][top]})"),f" | Allocated: {B0.sum()}/{c['boats']} boats, {T0.sum()}/{c['teams']} teams, {F0.sum()}/{c['food']} packets. Heat = population × expected severity × probability-weighted future multiplier, spread around each point (illustrative, not a flood model). Blue line thickness = region's share."+(f" {auto} region(s) had no coordinates and were placed automatically on land near Mumbai/Raigad." if auto else '')]
    bay=pd.DataFrame([{"Region":M['names'][i],"Reported":c['regions'][i]['sev'],**{f"P({l})":round(M['post'][i][k],2) for k,l in enumerate(LV)},"Expected severity (1-3)":round(M['sevE'][i],2)} for i in range(n)])
    nt=nash(M,c['greed'],c['cost']); st,lam=stack(M,c['aud'],c['infl'],c['pen']); rp,j=repeated(M,c['greed'],c['cost'],path)
    sh=pd.DataFrame({"Region":M['names'],"Shapley value":R['phi'].round(0),"Standalone v({i})":[round(R['V'][(i,)]) for i in range(n)]})
    games=html.Div([
     box("A. Bayesian game: beliefs about true severity",html.P("Reports are noisy; posterior = prior × likelihood given your reliability setting."),tbl(bay)),
     box("B. Non-cooperative game: cooperate vs selfish (pure Nash by exhaustive best-response check)",tbl(nt,style_data_conditional=[{'if':{'filter_query':'{Nash equilibrium} = "YES"'},'backgroundColor':'#ffe9a8'}])),
     box("C. Cooperative game: Shapley values (coalitions pool pro-rata endowments)",html.P(f"Grand-coalition value {R['V'][tuple(range(n))]:.0f} vs sum of standalone {sum(R['V'][(i,)] for i in range(n)):.0f}."),tbl(sh)),
     box("D. Stackelberg: authority commits to λ, regions then choose Honest/Inflate",html.P(f"Leader-optimal λ = {lam:.1f}"),tbl(st)),
     box("E. Repeated game over your round path",dcc.Graph(figure=px.line(rp,x='Round',y='Cumulative welfare',color='Strategy',markers=True)),
         html.P(f"Tempted region: {j}. Compare its own cumulative payoff in the 'defects' line vs 'All cooperate' in the table."),tbl(rp.round(0))),
     box("F. Minimax vs expected utility",tbl(R['cmp']))])
    B,T,Fd=rnd(c['boats'],f),rnd(c['teams'],f),rnd(c['food'],f); W,e,m=evalf(M,f)
    al=pd.DataFrame({"Region":M['names'],"Share":[f"{x:.0%}" for x in f],"Boats":B,"Teams":T,"Food packets":Fd})
    alloc=html.Div([box("Recommended allocation (Bayes + EU + Minimax + Shapley blend)",tbl(al),
        html.P(f"Expected welfare {e:.0f} | worst-case {m:.0f} | blend = (1-fair)×[(1-risk)×EU + risk×minimax] + fair×Shapley; this blend is a documented heuristic, not a theorem.")),
        dcc.Graph(figure=px.bar(al.melt('Region',['Boats','Teams','Food packets']),x='Region',y='value',color='variable',barmode='group'))])
    need=M['pop']*M['sevE']; Pm=float(M['P']@M['m'])
    t1=pd.DataFrame({"Future":M['fn'],"Entered p":[x['p'] for x in c['futures']],"Normalised p":M['P'].round(3),"Demand multiplier":M['m']})
    dm=pd.DataFrame({"Region":M['names'],"Weighted need = pop × E[severity]":need.round(0),"Boats needed":(need*Pm/M['cap'][0]).round(1),"Teams needed":(need*Pm/M['cap'][1]).round(1),"Packets needed":(M['pop']*Pm*M['cap'][2]).round(0)})
    sc=[("boats",dm['Boats needed'].sum(),c['boats']),("teams",dm['Teams needed'].sum(),c['teams']),("packets",dm['Packets needed'].sum(),c['food'])]
    bl=pd.DataFrame({"Region":M['names'],"EU-optimal share":R['fe'].round(3),"Minimax share":R['fm'].round(3),"Shapley share":R['shn'].round(3),"Final share":f.round(3)})
    trace=html.Div([
     box("Step 1. Inputs as understood",html.P(f"{n} regions, {len(M['m'])} futures. Probabilities rescaled to sum to 1. Coalition endowments use "+("your Own share % column." if any(x.get('own') for x in c['regions']) else "population proportions (no Own share % given).")),tbl(t1)),
     box("Step 2. Bayes",*[html.P(f"{M['names'][i]}: reported {c['regions'][i]['sev']}, prior {c['prior']}, likelihood of that report = {c['rel']} if true level matches, else {(1-c['rel'])/2:.3f} → posterior {M['post'][i].round(3).tolist()} → expected severity weight {M['sevE'][i]:.2f}") for i in range(n)]),
     box(f"Step 3. Demand vs supply at the probability-weighted multiplier ({Pm:.2f})",tbl(dm),html.P("Scarcity: "+"; ".join(f"{k} need {a:.1f} vs {b} available" for k,a,b in sc))),
     box("Step 4. Search performed",html.P(f"{R['nG']} candidate allocations scored against each of {len(M['m'])} futures (expected utility and worst-case); {2**n} cooperate/selfish profiles checked for Nash; {2**n} coalitions valued for Shapley; 11 values of λ tested for Stackelberg; {len(path)} rounds simulated.")),
     box("Step 5. Blend into recommendation",html.P(f"final = (1−{c['fair']})×[(1−{c['risk']})×EU + {c['risk']}×Minimax] + {c['fair']}×Shapley, then rounded to whole boats/teams/packets by largest remainder."),tbl(bl))])
    return fig,games,alloc,c,trace,'✔ Analysis complete: opened the Allocation tab.',info

@app.callback(Output('wiout','children'),Input('wi','n_clicks'),State('cfg','data'),*[State(i,'value') for i in('db','dt','dp','r2')],prevent_initial_call=True)
def whatif(_,c,db,dt,dp,r2):
    if not c: return html.P("Run the main analysis first.")
    c2=dict(c,boats=max(0,c['boats']+(db or 0)),teams=max(0,c['teams']+(dt or 0)),rel=r2); c2['futures']=[dict(x) for x in c['futures']]
    tot=sum(x['p'] for x in c2['futures']); c2['futures'][-1]['p']=max(0,c2['futures'][-1]['p']/tot*100+(dp or 0))/100*tot
    A,B=solve(c),solve(c2); rows=[]
    for i,n in enumerate(A['M']['names']):
        rows.append({"Region":n,"Share before":f"{A['f'][i]:.0%}","Share after":f"{B['f'][i]:.0%}",
          "Boats before→after":f"{rnd(c['boats'],A['f'])[i]}→{rnd(c2['boats'],B['f'])[i]}","Teams before→after":f"{rnd(c['teams'],A['f'])[i]}→{rnd(c2['teams'],B['f'])[i]}"})
    ea,eb=evalf(A['M'],A['f']),evalf(B['M'],B['f'])
    return html.Div([tbl(pd.DataFrame(rows)),html.P(f"Expected welfare {ea[1]:.0f} → {eb[1]:.0f} | worst-case {ea[2]:.0f} → {eb[2]:.0f}")])
if __name__=="__main__":
    import threading, webbrowser
    threading.Timer(1.5,lambda:webbrowser.open_new_tab("http://127.0.0.1:8050")).start()
    print("Dashboard at http://127.0.0.1:8050  (Ctrl+C to stop)")
    app.run(host="127.0.0.1",port=8050,debug=False)
