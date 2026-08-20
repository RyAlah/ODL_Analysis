import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, re, random
from wordcloud import WordCloud
from nltk.corpus import stopwords

SRC='/root/.claude/uploads/8233d335-e81d-51ef-8c70-4e86a921efcc/eb178790-feedback_comparative_periods.xlsx'
SURFACE='#000026'
# positive-leaning palette: deck blues + white + one green, no red
WORD_COLORS=['#4A85E8','#6B9FF2','#A5C9F7','#E7E7E7','#18A566','#307DE0']

df=pd.read_excel(SRC)
f=df[df['period']=='focus']
text=' '.join(f['What does Open Door Legal do well'].dropna().astype(str))

EN=set(stopwords.words('english'))
# Spanish stopwords (translation is offline here; the notebook version uses translated text)
ES={'que','muy','los','las','con','todo','para','por','del','una','uno','como','les','han',
    'son','fue','esta','este','mas','más','the','and','ellos','ella','porque','pero','sus',
    'nos','soy','estoy','muchas','mucho','bien','todos','todas','ser','hacer','tan','así',
    'me','le','lo','la','el','en','de','a','y','se','su','mi','es','un','no','na','personas','persona','ayudan','ayuda','ayudar','ayudaron','comunidad','siempre','gracias','abogado','abogada','trabajo','forma','muchisimo'}
ODL={'open','door','legal','odl','help','helped','helping','everything','idk','nothing',
     'thing','things','yes','also','would','could','really','well','good','great','service',
     'services','people','person','case','cases','get','got'}
STOP=EN|ES|ODL

text=re.sub(r'[^a-zA-Záéíóúñü\s]',' ',text.lower())
words=[w for w in text.split() if w not in STOP and len(w)>2]
freq=pd.Series(words).value_counts()
print('top 25 positive words (focus, raw do_well):')
print(freq.head(25).to_string())

def color_func(*a,**k):
    return random.choice(WORD_COLORS)

wc=WordCloud(width=1600,height=800,background_color=SURFACE,
             color_func=color_func,prefer_horizontal=0.9,
             max_words=90,min_font_size=12,relative_scaling=0.42,
             font_path=None,margin=6).generate_from_frequencies(freq.to_dict())

fig,ax=plt.subplots(figsize=(13,6.6),facecolor=SURFACE)
ax.imshow(wc,interpolation='bilinear'); ax.axis('off')
ax.set_title('What Clients Say ODL Does Well',color='#E7E7E7',
             fontweight='bold',fontsize=18,pad=14)
fig.savefig('/home/user/ODL_Analysis/outputs/slide_theme/fig_wordcloud_dowell.png',
            dpi=150,bbox_inches='tight',facecolor=SURFACE)
print('saved fig_wordcloud_dowell.png')
