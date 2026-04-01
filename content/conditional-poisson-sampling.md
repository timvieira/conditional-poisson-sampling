title: Conditional Poisson Sampling
date: 2026-03-25
comments: true
tags: notebook, sampling, algorithms, sampling-without-replacement

<macros>
\newcommand{\w}{w}
\newcommand{\bw}{\boldsymbol{w}}
\newcommand{\W}{W}
\newcommand{\Z}{Z}
\newcommand{\Zw}[2]{\binom{#1}{#2}}
\newcommand{\pip}{\pi}
\def\btheta{{\boldsymbol{\theta}}}
\def\bpip{\boldsymbol{\pi}}
\newcommand{\ba}{\boldsymbol{a}}
\newcommand{\bb}{\boldsymbol{b}}
\newcommand{\z}{z}
\newcommand{\llbracket}{[\![}
\newcommand{\rrbracket}{]\!]}
\newcommand{\defeq}{\overset{\small\text{def}}{=}}
</macros>

<div class="margin-note">
<a href="test_identities.py" class="verified" target="_blank">✓</a> = numerically verified—each links to its test case in <a href="test_identities.py" style="color:#999">test_identities.py</a>.
</div>

Suppose you want to draw a random subset of exactly $n$ items from a universe of $N$ items, where each item $i$ has a positive weight $\w_i$.<footnote>We assume $\w_i \in (0, \infty)$.  This is without loss of generality: items with $\w_i = 0$ are excluded from the universe (they are never selected), and items with $\w_i = \infty$ are deterministically included (reduce $n$ and $N$ by 1 each).  The general case reduces to the finite positive case by preprocessing.</footnote>
The **conditional Poisson distribution** assigns each size-$n$ subset a probability proportional to the product of its weights:<a href="test_identities.py#test_distribution_definition" title="test_distribution_definition, test_Z_is_elementary_symmetric_poly" class="verified" target="_blank">✓</a>

$$
P(S) \propto \mathbf{1}\big[ |S| = n\big] \prod_{i \in S} \w_i
$$

The normalizing constant $\Zw{\bw}{n} \defeq \sum_{|S|=n} \prod_{i \in S} \w_i$ is a weighted generalization of the binomial coefficient, which recovers $\binom{N}{n} = \Zw{\bw}{n}$ when $\bw = \mathbf{1}^N$.<a href="test_identities.py#test_Z_equals_binomial_when_uniform" title="test_Z_equals_binomial_when_uniform" class="verified" target="_blank">✓</a>

**Inclusion probabilities.** The inclusion probability $\pip_i \defeq \sum_{S} P(S)\, \mathbf{1}[i \in S]$.  Higher weight means higher inclusion probability, but the relationship is nonlinear because the other weights also matter—doubling $\w_i$ does not double $\pip_i$, as the other items push back through the size constraint $|S| = n$. Later in this article, we provide an interactive widget for exploring the nonlinear relationship between $\bw$ and $\bpip$.

**Why is this distribution special?** The conditional Poisson distribution is an exponential family with natural parameters $\theta_i \defeq \log \w_i$ and sufficient statistics $\mathbf{1}[i \in S]$.  Among all distributions over size-$n$ subsets with prescribed inclusion probabilities $\pip_i = P(i \in S)$, it is the unique *maximum-entropy* one<a href="test_identities.py#test_max_entropy" title="test_max_entropy" class="verified" target="_blank">✓</a>—making the fewest assumptions beyond the marginals ([Jaynes, 1957](https://doi.org/10.1103/PhysRev.106.620); [Chen, Dempster & Liu, 1994](https://academic.oup.com/biomet/article-abstract/81/3/457/256956)), in the same sense that the Gaussian is max-entropy for given mean and variance.  The log-normalizer $\log \Zw{\bw}{n}$ is convex in $\btheta$, so many properties follow mechanically: inclusion probabilities are the gradient ($\pip_i = \partial \log \Z / \partial \theta_i$) and fitting $\btheta$ to target inclusion probabilities is a convex optimization problem.  The distribution is also called the *exponential fixed-size design* for this reason.

**Relationship to Poisson sampling.** In Poisson sampling,<footnote>Named after mathematician [Siméon Denis Poisson](https://en.wikipedia.org/wiki/Sim%C3%A9on_Denis_Poisson).  Although poisson is the French word for fish, no fishing metaphor is intended.</footnote> each item $i$ is included independently with probability $p_i$, so the sample size $|S| = \sum_i \mathbf{1}[i \in S]$ is random.  The *conditional* Poisson distribution conditions on $|S| = n$ exactly—fixing the sample size while preserving the relative inclusion odds.  Under Poisson sampling, each item's inclusion probability is simply $\pip_i = p_i$; conditioning on $|S| = n$ makes $\pip_i$ depend on all the other weights too, which is what makes computing $\bpip$ nontrivial.  The weight $\w_i$ is the *odds* of the $i$<sup>th</sup> coin: $\w_i \defeq p_i / (1 - p_i)$, equivalently $p_i = \w_i/(1+\w_i)$.<a href="test_identities.py#test_weight_is_odds" title="test_weight_is_odds, test_conditional_poisson_from_bernoulli" class="verified" target="_blank">✓</a>  (The [identities section](#Parameterizations) compares the odds and probability parameterizations.)

**Sampling without replacement.** The following construction shows how conditional
Poisson can be used for sampling without replacement.  Draw $n$ items
independently from the categorical distribution over weights, and keep only the
samples where all draws are distinct:

<div class="pseudocode">
<b>repeat</b><br>
$\quad$ Draw $s_1, \ldots, s_n \overset{\text{i.i.d.}}{\sim} \text{Categorical}(\bw / \W)$ where $\W \defeq \sum_i \w_i$<br>
$\quad$ $S \leftarrow \{s_1, \ldots, s_n\}$<br>
<b>until</b> $|S| = n$ (all draws are distinct)<br>
<b>return</b> $S$
</div>

The resulting distribution over size-$n$ subsets is exactly $P(S)$.<a href="test_identities.py#test_rejection_bernoulli_produces_cps" title="test_rejection_bernoulli_produces_cps" class="verified" target="_blank">✓</a>
This rejection sampler is not a practical sampling algorithm;<footnote>The acceptance probability of this construction is $n! \cdot \Zw{\bw}{n} / \W^n$.<a href="test_identities.py#test_categorical_acceptance_rate" title="test_categorical_acceptance_rate" class="verified" target="_blank">✓</a></footnote>
it simply establishes what the distribution *is*.  We will derive sampling algorithms that sample from $P(S)$ efficiently.


**What this post covers.** The computational challenges are: computing $\Zw{\bw}{n}$ and $P(S)$, computing $\bpip$ from $\bw$, drawing exact samples $S \sim P$, and the inverse problem of finding $\bw$ from target $\bpip$.  This post gives efficient algorithms for all four—in $\mathcal{O}(N \log^2 n)$ time using a polynomial product tree.  The code is available as a [Python library](https://github.com/timvieira/conditional-poisson-sampling).

**Software.** As far as I can tell, this is the only publicly available library for conditional Poisson sampling in Python (or any language outside of R's survey-sampling packages).  Existing R implementations—`UPmaxentropy` in the [sampling](https://cran.r-project.org/web/packages/sampling/) package and the [BalancedSampling](https://cran.r-project.org/web/packages/BalancedSampling/) package—use either rejection sampling or $\mathcal{O}(Nn)$ dynamic programming.  The product-tree algorithm used here does not appear in any prior software that I'm aware of.


<style>
/* Tufte-style sidenote numbering */
article { counter-reset: sidenote-counter; }
.sidenote-number { counter-increment: sidenote-counter; }
.sidenote-number::after { content: counter(sidenote-counter); font-size: 0.6em; vertical-align: super; }
.sidenote-number + .margin-note::before { content: counter(sidenote-counter); font-size: 0.6em; vertical-align: super; }

#cps table { border-collapse: collapse; width: auto; margin: 0; }
#cps th, #cps td { padding: 1px 2px; font-family: inherit; font-size: 0.85em; line-height: 1.3; }
#cps th { border: none; font-weight: normal; color: #666; }
#cps td { border: none; }
#cps table { border: none; margin-bottom: 0.3em; }
#cps .rl { text-align: left; color: #999; font-size: 0.75em; }
#cps .ic { text-align: center; }
#cps .pc { text-align: right; }
#cps .zero { color: #ccc; }
#cps .bar-td {
  vertical-align: bottom; text-align: center; padding: 2px 1px;
  cursor: ns-resize; user-select: none; -webkit-user-select: none;
  touch-action: none;
}
#cps .bar-td.readonly { cursor: default; }
#cps .bar-td svg { display: block; margin: 0 auto; }
@media (max-width: 600px) {
  body { font-size: 14pt; padding: 0 0.5em; }
  #cps .rl { width: 60px; font-size: 0.75em; }
  #cps th, #cps td { padding: 1px 2px; }
}

</style>

<div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px 20px; margin: 16px 0;">

**Interactive explorer.** Drag the weight bars to see how changing $w_i$ affects the subset probabilities $P(S)$ and inclusion probabilities $\pi_i$. Drag the $\pi_i$ bars to solve the inverse problem: find weights that produce given inclusion probabilities. Use the $N$ and $n$ controls to change the problem size.

<div id="cps"></div>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
(function() {
  var N=5, n=3, w=[0.5,1.2,0.3,1.8,0.8], pi=[];
  var CW='#5b9bd5', CP='#c0504d', CI='#d4a24e';
  // Responsive bar sizing
  var mobile = window.innerWidth < 600;
  var barH = mobile ? 70 : 90;
  var bw = mobile ? 22 : 28;

  function subs(N,n){var r=[];(function go(s,c){if(c.length===n){r.push(c.slice());return;}if(s>=N)return;c.push(s);go(s+1,c);c.pop();go(s+1,c);})(0,[]);return r;}
  function getPi(w){
    var S=subs(N,n);
    var pr=S.map(function(s){return s.reduce(function(a,i){return a*w[i];},1);});
    var Z=pr.reduce(function(a,b){return a+b;},0);
    var p=Array(N).fill(0);
    if(Z>0) S.forEach(function(s,j){s.forEach(function(i){p[i]+=pr[j]/Z;});});
    return p;
  }
  function getTable(w){
    var S=subs(N,n);
    var pr=S.map(function(s){return s.reduce(function(a,i){return a*w[i];},1);});
    var Z=pr.reduce(function(a,b){return a+b;},0);
    return S.map(function(s,j){return{s:s,ind:Array.from({length:N},function(_,i){return s.indexOf(i)>=0?1:0;}),p:Z>0?pr[j]/Z:0};});
  }
  // Warm-start from current w, center θ to stabilize scale
  function fit(ps){
    var th=w.map(function(wi){return Math.log(Math.max(1e-10,wi));});
    for(var i=0;i<200;i++){
      var ww=th.map(function(t){return Math.exp(t);}),c=getPi(ww);
      var e=Math.max.apply(null,c.map(function(p,i){return Math.abs(p-ps[i]);}));
      if(e<1e-8)break;
      var g=c.map(function(p,i){return p-ps[i];}),s=1;
      for(var l=0;l<30;l++){
        var tn=th.map(function(t,i){return t-s*g[i];}),wn=tn.map(function(t){return Math.exp(t);});
        if(Math.max.apply(null,getPi(wn).map(function(p,i){return Math.abs(p-ps[i]);}))<e){th=tn;break;}
        s*=0.5;
      }
    }
    // Center log-weights so geometric mean = 1 (scale-invariant, keeps values moderate)
    var mean = th.reduce(function(a,b){return a+b;},0) / th.length;
    return th.map(function(t){return Math.exp(t - mean);});
  }

  var WMAX = 3;
  function getMaxW(){ return WMAX; }

  var root=d3.select('#cps');
  var wSvgs=[], wFills=[], wLabels=[];
  var pSvgs=[], pFills=[], pLabels=[];
  var probCells=[];

  function makeBar(td, val, maxVal, color, draggable, onDrag) {
    var svg = td.append('svg').attr('width',bw).attr('height',barH);
    svg.append('rect').attr('x',0).attr('y',0).attr('width',bw).attr('height',barH)
      .attr('fill','#f8f8f8').attr('stroke','#eee').attr('rx',2);
    var frac = Math.min(val/maxVal, 1);
    var fill = svg.append('rect').attr('x',1).attr('width',bw-2).attr('rx',2)
      .attr('y',barH-frac*barH).attr('height',frac*barH)
      .attr('fill',color).attr('opacity',0.7).style('pointer-events','none');
    var label = svg.append('text').attr('x',bw/2).attr('y',barH-frac*barH-4)
      .attr('text-anchor','middle').style('font-size','11px').style('fill',color).style('font-family',"'EB Garamond', serif")
      .style('pointer-events','none').text(val === 0 ? '0' : val < 0.01 ? '\u22480' : val.toFixed(3));
    if (draggable) {
      svg.append('rect').attr('width',bw).attr('height',barH)
        .attr('fill','transparent').attr('cursor','ns-resize')
        .style('touch-action','none')
        .call(d3.drag().on('drag',function(event){
          var frac = (barH - event.y) / barH;
          onDrag(Math.max(0, Math.min(1, frac)));
        }));
    }
    return {fill:fill, label:label};
  }

  function build() {
    root.selectAll('*').remove();
    wSvgs=[]; wFills=[]; wLabels=[];
    pSvgs=[]; pFills=[]; pLabels=[];
    probCells=[];
    pi = getPi(w);

    // Controls
    var ctrl = root.append('div').style('font-size','0.9em').style('margin-bottom','8px').style('font-family','inherit');
    ctrl.append('span').html('$N$ = ');
    ctrl.append('input').attr('type','number').attr('min',2).attr('max',8).attr('value',N)
      .style('width','44px').style('font-family','inherit').style('font-size','inherit')
      .style('border','1px solid #ccc').style('border-radius','3px').style('padding','2px 4px')
      .on('change input',function(){var v=+this.value;if(isNaN(v))return;if(v<2||v>8){d3.select('#cps-status').text('the distribution requires 2 \u2264 N \u2264 8').style('color','#c00');return;}v=Math.round(v);if(v===N)return;N=v;n=Math.min(n,N);while(w.length<N)w.push(0.5+Math.random());w=w.slice(0,N);build();});
    ctrl.append('span').html('&ensp;$n$ = ');
    ctrl.append('input').attr('type','number').attr('min',0).attr('max',N).attr('value',n)
      .style('width','44px').style('font-family','inherit').style('font-size','inherit')
      .style('border','1px solid #ccc').style('border-radius','3px').style('padding','2px 4px')
      .on('change input',function(){var v=+this.value;if(isNaN(v))return;if(v<0||v>N){d3.select('#cps-status').text('the distribution requires 0 \u2264 n \u2264 N='+N).style('color','#c00');return;}v=Math.round(v);if(v===n)return;n=v;build();});

    // === ONE TABLE for everything ===
    var tbl = root.append('table');
    var cg = tbl.append('colgroup');
    cg.append('col').style('width', 'auto');
    for (var j=0;j<N;j++) cg.append('col').style('width', (bw+8)+'px');
    cg.append('col').style('width', '55px');
    var tbody = tbl.append('tbody');

    // --- Weight labels ---
    var wh = tbody.append('tr');
    wh.append('td').attr('class','rl');
    for(var i=0;i<N;i++) wh.append('td').attr('class','ic').style('color',CI).html('$w_'+(i+1)+'$');
    wh.append('td').attr('class','pc');

    // --- Weight bars ---
    var wb = tbody.append('tr');
    wb.append('td').attr('class','rl');
    for(var i=0;i<N;i++){
      (function(idx){
        var td = wb.append('td').attr('class','bar-td');
        var b = makeBar(td, w[idx], WMAX, CI, true, function(frac){
          w[idx] = Math.max(0.01, frac * WMAX);
          pi = getPi(w);
          update();
        });
        wFills.push(b.fill); wLabels.push(b.label);
      })(i);
    }
    wb.append('td').attr('class','pc');

    // --- Spacer ---
    tbody.append('tr').append('td').attr('colspan',N+2).style('height','6px');

    // --- Subsets header ---
    var allS = subs(N,n);
    probCells = [];
    if (allS.length <= 70) {
      var tdata = getTable(w);
      var sh = tbody.append('tr');
      sh.append('td').attr('class','rl').style('text-align','right').html('Subset $S$');
      for(var j=0;j<N;j++) sh.append('td').attr('class','ic');
      sh.append('td').attr('class','pc').style('text-align','left').style('color',CW).html('$P(S)$');

      // --- Subset rows ---
      tdata.forEach(function(r){
        var tr = tbody.append('tr');
        tr.append('td').attr('class','rl').style('color','#333').style('font-style','normal').style('text-align','right').text('{'+r.s.map(function(j){return j+1;}).join(', ')+'}');
        r.ind.forEach(function(v){tr.append('td').attr('class','ic'+(v?'':' zero')).text(v);});
        var pc = tr.append('td').attr('class','pc').style('text-align','left').style('padding','1px 2px');
        var barWrap = pc.append('div').style('display','flex').style('align-items','center').style('gap','3px');
        barWrap.append('div').attr('class','prob-bar')
          .style('height','14px').style('border-radius','2px')
          .style('background',CW).style('opacity','0.7')
          .style('min-width','1px');
        barWrap.append('span').attr('class','prob-val')
          .style('font-size','0.8em').style('color',CW).style('white-space','nowrap');
        probCells.push(pc);
      });
    } else {
      tbody.append('tr').append('td').attr('colspan',N+2).style('color','#999').style('font-size','0.85em').text('('+allS.length+' subsets\u2014table hidden)');
    }

    // --- Spacer ---
    tbody.append('tr').append('td').attr('colspan',N+2).style('height','6px');

    // --- Pi header ---
    var ph = tbody.append('tr');
    ph.append('td').attr('class','rl').style('color',CP).html('inclusion prob.');
    for(var i=0;i<N;i++) ph.append('td').attr('class','ic').style('color',CP).style('font-weight','bold').html('$\\pi_'+(i+1)+'$');
    ph.append('td').attr('class','pc');

    // --- Pi bars ---
    var pb = tbody.append('tr');
    pb.append('td').attr('class','rl').style('font-size','0.7em').style('color','#999').html('drag to set target<br>($\\sum \\pi_i = n$)');
    for(var i=0;i<N;i++){
      (function(idx){
        var td = pb.append('td').attr('class','bar-td');
        var b = makeBar(td, pi[idx], 1, CP, true, function(frac){
          // Set this pi as target, solve for w
          pi[idx] = Math.max(0.02, Math.min(0.98, frac));
          w = fit(pi);
          WMAX = Math.max(3, Math.max.apply(null, w) * 1.2);
          pi = getPi(w);
          update();
        });
        pFills.push(b.fill); pLabels.push(b.label);
      })(i);
    }
    pb.append('td').attr('class','pc');

    update();
    // Typeset MathJax after DOM is rebuilt
    if (window.MathJax && MathJax.typesetPromise) {
      MathJax.typesetClear();
      MathJax.typesetPromise();
    }
  }

  function update() {
    var maxW = getMaxW();
    for(var i=0;i<N;i++){
      var wFrac=Math.min(w[i]/maxW,1);
      wFills[i].attr('y',barH-wFrac*barH).attr('height',wFrac*barH);
      wLabels[i].attr('y',barH-wFrac*barH-4).text(w[i] === 0 ? '0' : w[i] < 0.01 ? '\u22480' : w[i].toFixed(3));
      var pFrac=pi[i];
      pFills[i].attr('y',barH-pFrac*barH).attr('height',pFrac*barH);
      pLabels[i].attr('y',barH-pFrac*barH-4).text(pi[i] === 0 ? '0' : pi[i] < 0.01 ? '\u22480' : pi[i].toFixed(3));
    }
    if(probCells.length>0){
      var tdata=getTable(w);
      var maxP=Math.max.apply(null,tdata.map(function(r){return r.p;}));
      tdata.forEach(function(r,i){
        if(!probCells[i])return;
        probCells[i].select('.prob-val').text(r.p.toFixed(3));
        probCells[i].select('.prob-bar').style('width', Math.max(1, r.p/maxP*60)+'px');
      });
    }
  }

  build();
})();
</script>

</div>


## The Polynomial Product Tree

The key idea is to encode the sum over all $\binom{N}{n}$ subsets as the coefficient of $\z^n$ in a product of polynomials:

$$
(1 + \w_1 \z)(1 + \w_2 \z) \cdots (1 + \w_N \z) = \sum_{k=0}^{N} \Zw{\bw}{k}\, \z^k
$$

The $n$<sup>th</sup> coefficient is exactly $\Zw{\bw}{n}$, the normalizing constant.<a href="test_identities.py#test_product_polynomial_coefficients" title="test_product_polynomial_coefficients" class="verified" target="_blank">✓</a>  This product can be computed in $\mathcal{O}(N \log^2 n)$ time using a divide-and-conquer strategy on a binary tree—a standard technique from computer algebra known as the *subproduct tree* (see [von zur Gathen & Gerhard (2013)](https://doi.org/10.1017/CBO9781139856065), Chapter 10).

**Polynomials as arrays.** Each polynomial is stored as an array of coefficients: `[1, 3, 2]` represents $1 + 3\z + 2\z^2$.  Multiplying two polynomials is *convolution* of their coefficient arrays—for example, $(1 + 2\z)(1 + 3\z)$ corresponds to convolving `[1, 2]` with `[1, 3]` to get `[1, 5, 6]`.  Using the fast Fourier transform (FFT), this convolution costs $\mathcal{O}(d \log d)$ where $d$ is the degree, rather than $\mathcal{O}(d^2)$ for the schoolbook method.  This is the key fact that makes the product tree $\mathcal{O}(N \log^2 n)$ instead of $\mathcal{O}(Nn)$.

**Notation.** We write $\llbracket f \rrbracket(\z^k)$ for the coefficient of $\z^k$ in a formal power series $f(\z) = \sum_k a_k \z^k$, i.e., $\llbracket f \rrbracket(\z^k) = a_k$.  This is sometimes written $[\z^k]\, f(\z)$; we use the Scott bracket notation to avoid ambiguity with other uses of square brackets.

### Computing the Normalizing Constant $\Z$

Each leaf of a complete binary tree holds one degree-1 polynomial $(1 + \w_i \z)$.  Internal nodes multiply their children's polynomials.  The root holds the full product, whose $n$<sup>th</sup> coefficient is $\Zw{\bw}{n}$.

<style>
#content { overflow: visible !important; }
svg text { font-family: 'EB Garamond', serif; }
#tw-controls {
  display: flex; align-items: center; gap: 12px;
  margin-bottom: 8px; font-size: 0.9em;
}
#tw-controls input[type=number] {
  width: 44px; padding: 2px 4px;
  font-family: inherit; font-size: inherit;
  border: 1px solid #ccc; border-radius: 3px;
}
#tw-status { font-size: 0.75em; color: #999; margin-top: 4px; }
</style>

<div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px 20px; margin: 16px 0; width: fit-content; min-width: 100%; max-width: none;">

**Interactive product tree.** Drag the weight sliders to see how changing $w_i$ affects the polynomial coefficients at every node. The tree builds the product $\prod_i(1 + w_i z)$ bottom-up; the $n$<sup>th</sup> coefficient at the root (highlighted in red) is the normalizing constant $Z$. Use the $N$ and $n$ controls to change the problem size.

<div id="tw"></div>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
(function() {
  var N = 4, n = 2;
  var w = [0.5, 1.2, 0.3, 1.0];
  var WMAX = 3;

  var CW = '#5b9bd5', CR = '#c0504d', CI = '#d4a24e'; // blue, red, input gold
  var root = d3.select('#tw');

  function polyMul(a, b) {
    var out = new Array(a.length + b.length - 1).fill(0);
    for (var i = 0; i < a.length; i++)
      for (var j = 0; j < b.length; j++)
        out[i+j] += a[i] * b[j];
    return out;
  }

  function buildTree(w, n) {
    var leaves = w.map(function(wi, i) {
      return { poly: [1, wi], items: [i], leaf: true };
    });
    var size = 1;
    while (size < leaves.length) size *= 2;
    while (leaves.length < size)
      leaves.push({ poly: [1], items: [], leaf: true, pad: true });
    var level = leaves;
    var allLevels = [level];
    while (level.length > 1) {
      var next = [];
      for (var i = 0; i < level.length; i += 2) {
        var l = level[i], r = level[i+1];
        var p = polyMul(l.poly, r.poly);
        if (p.length > n + 1) p = p.slice(0, n + 1);
        next.push({ left: l, right: r, poly: p, items: l.items.concat(r.items), leaf: false });
      }
      level = next;
      allLevels.push(level);
    }
    return { root: level[0], levels: allLevels };
  }

  // Each histogram bar width
  // Bar dimensions — same as cps-widget sliders
  var mobile = window.innerWidth < 600;
  var bW = mobile ? 22 : 28;  // bar width (same as slider)
  var bH = mobile ? 50 : 70;  // bar height (same as slider)
  var sliderH = bH;            // sliders are the same size
  var bGap = 4;                // gap between bars within a node
  var nodePad = 12;            // padding inside group box
  var vGap = 30;
  var hGap = 20;

  var sliderFills = [], sliderLabels = [];
  var nodeRefs = [];
  var zLabelDiv = null;

  // Place MathJax labels as absolutely-positioned HTML divs over the SVG
  // svgContainer must be position:relative
  var mathLabels = [];
  function addMathLabel(container, x, y, html, opts) {
    opts = opts || {};
    var div = container.append('div')
      .style('position', 'absolute')
      .style('left', x + 'px')
      .style('top', y + 'px')
      .style('font-size', opts.fontSize || '10px')
      .style('color', opts.color || '#999')
      .style('pointer-events', 'none')
      .style('white-space', 'nowrap');
    if (opts.anchor === 'middle') {
      div.style('transform', 'translateX(-50%)');
    } else if (opts.anchor === 'end') {
      div.style('transform', 'translateX(-100%)');
    }
    div.html(html);
    mathLabels.push(div);
    return div;
  }

  var dur = 400;
  var animating = false;
  var lastAction = null; // 'add' or 'remove' or null

  function changeN(target) {
    if (animating || target === N) return;
    if (target < 2) return;
    animating = true;

    function step() {
      if (N === target) { animating = false; return; }
      if (target > N) {
        w.push(0.5 + Math.random());
        N = N + 1;
        lastAction = 'add';
      } else {
        w.pop();
        N = N - 1;
        lastAction = 'remove';
      }
      n = Math.min(n, N);
      buildInner();
      if (N !== target) {
        setTimeout(step, dur);
      } else {
        animating = false;
      }
    }
    step();
  }

  function build() {
    buildInner();
  }

  function buildInner() {
    root.selectAll('*').remove();
    sliderFills = []; sliderLabels = [];
    nodeRefs = [];
    mathLabels = [];

    // Controls
    var ctrl = root.append('div').attr('id', 'tw-controls');
    ctrl.append('span').html('$N$ = ');
    ctrl.append('input').attr('type','number').attr('min',2).attr('value',N)
      .on('change input', function() {
        var v = Math.max(2, Math.round(+this.value));
        if (v === N) return;
        changeN(v);
      });
    ctrl.append('span').html('&ensp;$n$ = ');
    ctrl.append('input').attr('type','number').attr('min',0).attr('max',N).attr('value',n)
      .on('change input', function() {
        var v = Math.max(0, Math.min(N, Math.round(+this.value)));
        if (v === n) return;
        n = v; build();
      });

    var tree = buildTree(w, n);
    var levels = tree.levels;
    var depth = levels.length;

    // Layout: leaves at top, root at bottom
    // Compute each node's box width
    function nodeBoxW(nd) {
      if (nd.pad) return 0;
      var nc = Math.min(nd.poly.length, n + 1);
      return nc * (bW + bGap) - bGap + nodePad;
    }

    var levelSpacings = levels.map(function(lev) {
      var maxW = 0;
      lev.forEach(function(nd) { var w = nodeBoxW(nd); if (w > maxW) maxW = w; });
      return maxW + hGap;
    });
    levelSpacings[0] = Math.max(levelSpacings[0], bW + hGap + 10);

    // Find the widest level to set SVG width
    var maxLevelW = 0;
    for (var li = 0; li < levels.length; li++) {
      var lw = levels[li].length * levelSpacings[li];
      if (lw > maxLevelW) maxLevelW = lw;
    }
    var leafCount = levels[0].length;
    var svgW = Math.max(300, leafCount * levelSpacings[0] + 60);

    var labelH = 16;
    var topPad = 10;
    var sepGap = 20;       // space around separator lines
    var arrowLen = 14;     // connector arrows between zones
    var nodeH = bH + nodePad;

    // Zone 1: inputs (labels + sliders)
    var inputZoneTop = topPad;
    var inputZoneBot = topPad + labelH + sliderH + 4;
    // Separator 1
    var sep1Y = inputZoneBot + sepGap/2;
    // Zone 2: circuit (tree nodes)
    var leafY = sep1Y + sepGap/2 + arrowLen;
    var rootBot = leafY + (depth - 1) * (nodeH + vGap) + nodeH;
    // Separator 2
    var sep2Y = rootBot + sepGap;
    // Zone 3: output (Z label)
    var outputY = sep2Y + sepGap/2 + arrowLen;
    var svgH = outputY + 18;

    root.style('width', svgW + 'px');
    var svgWrap = root.append('div')
      .style('position', 'relative')
      .style('width', svgW + 'px')
      .style('height', svgH + 'px');
    var svg = svgWrap.append('svg')
      .attr('width', svgW).attr('height', svgH)
      .style('display', 'block').style('user-select', 'none');

    // Assign x: leaves left-aligned, so adding on the right doesn't shift the left
    var leafSpacing = levelSpacings[0];
    var leafStartX = 20;
    for (var ni = 0; ni < leafCount; ni++) {
      levels[0][ni].x = leafStartX + ni * leafSpacing + leafSpacing / 2;
      levels[0][ni].y = leafY;
    }
    // Internal levels: x = midpoint of children
    for (var li = 1; li < levels.length; li++) {
      levels[li].forEach(function(nd) {
        if (nd.left && nd.right) {
          nd.x = (nd.left.x + nd.right.x) / 2;
        } else if (nd.left) {
          nd.x = nd.left.x;
        }
        nd.y = leafY + li * (nodeH + vGap);
      });
    }

    // --- Zone separators and connectors ---
    // Separator 1: inputs → circuit
    svg.append('line').attr('x1', 10).attr('x2', svgW - 10)
      .attr('y1', sep1Y).attr('y2', sep1Y)
      .attr('stroke', '#e0e0e0').attr('stroke-width', 1).attr('stroke-dasharray', '4,4');

    // Arrow markers
    var arrowDef = svg.append('defs');
    // Arrowhead pointing along line direction (right-pointing triangle, orient=auto rotates it)
    arrowDef.append('marker').attr('id','arrowDown').attr('viewBox','0 0 10 10')
      .attr('refX',10).attr('refY',5)
      .attr('markerWidth',7).attr('markerHeight',7).attr('orient','auto')
      .append('path').attr('d','M0,0 L10,5 L0,10 Z').attr('fill','#ccc');

    // Connector lines drawn after nodes (need _barXs), see below

    // Separator 2: circuit → output
    svg.append('line').attr('x1', 10).attr('x2', svgW - 10)
      .attr('y1', sep2Y).attr('y2', sep2Y)
      .attr('stroke', '#e0e0e0').attr('stroke-width', 1).attr('stroke-dasharray', '4,4');

    // Arrow from root's n-th bar bottom to output zone (drawn after nodes so _barXs exists)
    arrowDef.append('marker').attr('id','arrowDownRed').attr('viewBox','0 0 10 10')
      .attr('refX',10).attr('refY',5)
      .attr('markerWidth',8).attr('markerHeight',8).attr('orient','auto')
      .append('path').attr('d','M0,0 L10,5 L0,10 Z').attr('fill', CR);
    // (actual line drawn after nodes, see below)

    // --- Draw edges ---
    for (var li = 1; li < levels.length; li++) {
      levels[li].forEach(function(nd) {
        [nd.left, nd.right].forEach(function(child) {
          if (!child || child.pad) return;
          var x1 = child.x, y1 = child.y + bH + nodePad;
          var x2 = nd.x, y2 = nd.y;
          var my = (y1 + y2) / 2;
          // Only tag edge if the child is the new leaf
          var edgeNew = lastAction === 'add' && child.leaf && child.items && child.items.indexOf(N-1) >= 0;
          svg.append('path')
            .attr('d', 'M'+x1+','+y1+' C'+x1+','+my+' '+x2+','+my+' '+x2+','+y2)
            .attr('fill', 'none').attr('stroke', '#ddd').attr('stroke-width', 1.2)
            .classed('new-node', edgeNew);
        });
      });
    }

    // --- Draw histograms for all nodes (log scale, fixed reference) ---
    var logMax = Math.log1p(Math.pow(WMAX, n));
    for (var li = 0; li < levels.length; li++) {
      levels[li].forEach(function(nd) {
        if (nd.pad) { nodeRefs.push(null); return; }
        var coeffs = nd.poly;
        var nCoeffs = Math.min(coeffs.length, n + 1);
        var boxW = nCoeffs * (bW + bGap) - bGap + nodePad;
        var gx = nd.x - boxW/2;
        // Only the leaf itself is "new" — ancestors already existed
        var isNew = lastAction === 'add' && nd.leaf && nd.items && nd.items.indexOf(N-1) >= 0;
        var g = svg.append('g')
          .attr('transform', 'translate(' + gx + ',' + nd.y + ')')
          .classed('new-node', isNew);

        // Group box
        g.append('rect').attr('width', boxW).attr('height', bH + nodePad)
          .attr('fill', '#f9f9f9').attr('stroke', 'none').attr('rx', 5);
        // Clip to keep labels inside
        var clipId = 'clip-' + li + '-' + ni;
        svg.append('defs').append('clipPath').attr('id', clipId)
          .append('rect').attr('width', boxW + 4).attr('height', bH + nodePad + 16)
          .attr('x', -2).attr('y', -14);
        g.attr('clip-path', 'url(#' + clipId + ')');

        var cRects = [], cTexts = [];
        for (var k = 0; k < nCoeffs; k++) {
          var bx = nodePad/2 + k * (bW + bGap);
          var barFrac = Math.min(1, Math.log1p(Math.abs(coeffs[k])) / logMax);
          var barPx = Math.max(0.5, barFrac * bH);
          var isHighlight = (li === levels.length - 1 && k === n);
          var col = isHighlight ? CR : CW;
          var by = nodePad/2; // vertical offset inside group
          // Track (same as cps-widget slider)
          g.append('rect')
            .attr('x', bx).attr('y', by)
            .attr('width', bW).attr('height', bH)
            .attr('fill', '#f8f8f8').attr('stroke', '#eee').attr('rx', 2);
          // Fill
          var cr = g.append('rect')
            .attr('x', bx + 1).attr('width', bW - 2).attr('rx', 2)
            .attr('y', by + bH - barPx).attr('height', barPx)
            .attr('fill', col).attr('opacity', 0.7);
          cRects.push(cr);
          // Value label above fill
          var ct = g.append('text')
            .attr('x', bx + bW/2).attr('y', nodePad/2 + bH - barPx - 3)
            .attr('text-anchor', 'middle')
            .style('font-size', '9px').style('fill', col)
            .style('font-family', "'EB Garamond', serif")
            .text(fmtCoeff(coeffs[k]));
          cTexts.push(ct);
        }

        // At root: add index labels below
        if (li === levels.length - 1) {
          for (var k = 0; k < nCoeffs; k++) {
            var labelX = gx + nodePad/2 + k * (bW + bGap) + bW/2;
            addMathLabel(svgWrap, labelX, nd.y + bH + 4,
              k === n ? '$\\mathbf{' + k + '}$' : '$' + k + '$',
              {anchor:'middle', color: k === n ? CR : '#bbb', fontSize:'9px'});
          }
        }

        // Store x positions for slider alignment
        nd._barXs = [];
        for (var k = 0; k < nCoeffs; k++) {
          nd._barXs.push(gx + nodePad/2 + k * (bW + bGap));
        }

        nodeRefs.push({ cellRects: cRects, cellTexts: cTexts, nCoeffs: nCoeffs });
      });
    }

    // --- Root → output arrow (from n-th bar center) ---
    var rootNdFinal = levels[levels.length - 1][0];
    var zLabelX = rootNdFinal.x; // default
    if (rootNdFinal._barXs && n < rootNdFinal._barXs.length) {
      var rootCx = rootNdFinal._barXs[n] + bW/2;
      zLabelX = rootCx;
      // Bezier curve from red bar to Z label, with manual arrowhead
      var x1 = rootCx, y1 = rootBot + 8;
      var x2 = rootNdFinal.x, y2 = outputY;
      zLabelX = x2;
      // Control points: leave downward, arrive downward
      var cp1x = x1, cp1y = y1 + (y2-y1)*0.7;
      var cp2x = x2, cp2y = y1 + (y2-y1)*0.3;
      // Shorten the path so arrowhead doesn't overshoot
      var tipGap = 6;
      var y2s = y2 - tipGap;
      svg.append('path')
        .attr('d', 'M'+x1+','+y1+' C'+cp1x+','+cp1y+' '+cp2x+','+cp2y+' '+x2+','+y2s)
        .attr('fill', 'none').attr('stroke', '#d4a0a0').attr('stroke-width', 1.2);
      // Manual arrowhead: triangle at endpoint, oriented along tangent from cp2 to endpoint
      var dx = x2 - cp2x, dy = y2s - cp2y;
      var len = Math.sqrt(dx*dx + dy*dy);
      if (len > 0) { dx /= len; dy /= len; }
      var ax = 5; // arrowhead half-width
      var al = 8; // arrowhead length
      // Tip at (x2, y2), base perpendicular to tangent
      var bx = x2 - dx*al, by = y2 - dy*al;
      var px = -dy, py = dx; // perpendicular
      svg.append('path')
        .attr('d', 'M'+x2+','+y2+
          ' L'+(bx+px*ax)+','+(by+py*ax)+
          ' L'+(bx-px*ax)+','+(by-py*ax)+' Z')
        .attr('fill', '#d4a0a0');
    }

    // --- Connector arrows from slider bottom to leaf w_i bar top ---
    levels[0].forEach(function(nd, idx) {
      if (nd.pad || !nd._barXs) return;
      var cx = nd._barXs[1] + bW/2;
      var isNew = lastAction === 'add' && nd.items && nd.items.indexOf(N-1) >= 0 && nd.leaf;
      svg.append('line')
        .attr('x1', cx).attr('y1', inputZoneBot)
        .attr('x2', cx).attr('y2', nd.y)
        .attr('stroke', '#ccc').attr('stroke-width', 1)
        .attr('marker-end', 'url(#arrowDown)')
        .classed('new-node', isNew);
    });

    // --- Sliders above leaves, aligned to the w_i bar (index 1) ---
    levels[0].forEach(function(nd, idx) {
      if (nd.pad) return;
      (function(idx) {
        var barX = nd._barXs[1];
        var sliderTop = topPad + labelH;
        var isNew = lastAction === 'add' && idx === N - 1;
        var sg = svg.append('g').classed('new-node', isNew);

        // Track
        sg.append('rect')
          .attr('x', barX + 1).attr('y', sliderTop)
          .attr('width', bW - 2).attr('height', sliderH)
          .attr('fill', '#f8f8f8').attr('stroke', '#eee').attr('rx', 1);

        // Fill
        var frac = Math.min(w[idx] / WMAX, 1);
        var sf = sg.append('rect')
          .attr('x', barX + 2).attr('width', bW - 4).attr('rx', 1)
          .attr('y', sliderTop + sliderH - frac * sliderH).attr('height', frac * sliderH)
          .attr('fill', CI).attr('opacity', 0.8).style('pointer-events', 'none');
        sliderFills.push(sf);

        // Label
        var sl = sg.append('text')
          .attr('x', barX + bW/2).attr('y', sliderTop + sliderH - frac * sliderH - 2)
          .attr('text-anchor', 'middle')
          .style('font-size', '9px').style('fill', CI).style('pointer-events', 'none')
          .text(w[idx].toFixed(2));
        sliderLabels.push(sl);

        // Item label above slider (MathJax)
        var ml = addMathLabel(svgWrap, barX + bW/2, topPad, '$w_{' + (idx+1) + '}$', {anchor:'middle', color:CI, fontSize:'11px'});
        if (isNew) ml.classed('new-node', true);

        // Drag target (wider for touch)
        sg.append('rect')
          .attr('x', barX - 4).attr('y', sliderTop - 2)
          .attr('width', bW + 8).attr('height', sliderH + 4)
          .attr('fill', 'transparent').attr('cursor', 'ns-resize')
          .style('touch-action', 'none')
          .call(d3.drag().on('drag', function(event) {
            var frac = (sliderTop + sliderH - event.y) / sliderH;
            w[idx] = Math.max(0.01, Math.min(WMAX, frac * WMAX));
            updateTree();
          }));
      })(idx);
    });

    // Z label in output zone, centered on the n-th bar
    zLabelDiv = addMathLabel(svgWrap, zLabelX, outputY, '', {anchor:'middle', color: CR, fontSize: '14px'});

    root.datum({ levels: levels, tree: tree });
    updateTree();

    // Animate new node sprouting in: fade in elements tagged 'new'
    if (lastAction === 'add') {
      svg.selectAll('.new-node')
        .style('opacity', 0)
        .transition().duration(dur)
        .style('opacity', null);
      mathLabels.forEach(function(ml) {
        if (ml.classed('new-node')) {
          ml.style('opacity', 0).transition().duration(dur).style('opacity', '1');
        }
      });
    }
    lastAction = null;

    // MathJax: typeset after fade starts
    setTimeout(function() {
      if (window.MathJax && MathJax.typesetPromise) {
        var tw = document.getElementById('tw');
        MathJax.typesetClear([tw]);
        MathJax.typesetPromise([tw]);
      }
    }, 10);
  }

  function fmtCoeff(v) {
    if (v === 0) return '0';
    if (v < 0.005 && v > 0) return '';
    if (v >= 10) return v.toFixed(0);
    if (v >= 1) return v.toFixed(1);
    return v.toFixed(2);
  }

  function updateTree() {
    var data = root.datum();
    var levels = data.levels;
    var sliderTop = 10 + 16; // topPad + labelH

    // Update sliders
    for (var i = 0; i < N; i++) {
      var frac = Math.min(w[i] / WMAX, 1);
      sliderFills[i].attr('y', sliderTop + sliderH - frac * sliderH).attr('height', frac * sliderH);
      sliderLabels[i].attr('y', sliderTop + sliderH - frac * sliderH - 2).text(w[i].toFixed(2));
    }

    // Rebuild tree data
    var newTree = buildTree(w, n);
    var newLevels = newTree.levels;

    // Update histograms (log scale, fixed reference)
    var logMax = Math.log1p(Math.pow(WMAX, n));
    var ri = 0;
    for (var li = 0; li < newLevels.length; li++) {
      for (var ni = 0; ni < newLevels[li].length; ni++) {
        var ref = nodeRefs[ri++];
        if (!ref) continue;
        var coeffs = newLevels[li][ni].poly;
        var nCoeffs = ref.nCoeffs;
        for (var k = 0; k < nCoeffs; k++) {
          var barFrac = Math.min(1, Math.log1p(Math.abs(coeffs[k])) / logMax);
          var barPx = Math.max(1, barFrac * bH);
          ref.cellRects[k].attr('y', nodePad/2 + bH - barPx).attr('height', barPx);
          if (ref.cellTexts[k]) {
            ref.cellTexts[k].attr('y', nodePad/2 + bH - barPx - 3).text(fmtCoeff(coeffs[k]));
          }
        }
      }
    }

    var Z = newTree.root.poly[Math.min(n, newTree.root.poly.length - 1)];
    var zNode = zLabelDiv.node();
    zNode.innerHTML = '$Z = ' + (Z !== undefined ? Z.toFixed(4) : '\\text{—}') + '$';
    if (window.MathJax && MathJax.typesetPromise) {
      MathJax.typesetClear([zNode]);
      MathJax.typesetPromise([zNode]);
    }
  }

  build();
})();
</script>

</div>


**Complexity.** At each level of the tree, the total size of the polynomials being multiplied is $\mathcal{O}(N)$, and each multiplication is done via FFT in $\mathcal{O}(d \log d)$ time where $d$ is the degree. The recurrence is

$$T(N) = 2\,T(N/2) + \mathcal{O}(N \log N), \quad T(1) = \mathcal{O}(1)$$

which solves to $T(N) = \mathcal{O}(N \log^2 n)$ by the Master Theorem.
The complexity is $\mathcal{O}(N \log^2 n)$ rather than $\mathcal{O}(N \log^2 N)$ because we only need the coefficient at $\z^n$: intermediate polynomials can be **truncated to degree $n$** without affecting the result.  (Convolution of two polynomials truncated to degree $n$ still gives the correct coefficients up to degree $n$.)  This reduces both the polynomial sizes and the FFT costs throughout the tree.


### Computing Inclusion Probabilities $\bpip$

The inclusion probability is the gradient of the log-normalizer (an exponential family identity):

$$\pip_i \defeq \frac{\partial \log \Zw{\bw}{n}}{\partial \theta_i}$$

Since the product tree already computes $\log \Zw{\bw}{n}$, we get $\bpip$ by running [backpropagation](https://timvieira.github.io/blog/evaluating-fx-is-as-fast-as-fx/) on the same tree—no new algorithm needed.  By the Baur-Strassen theorem ([1983](https://doi.org/10.1016/0304-3975(83)90110-X)), the gradient costs at most a constant factor more than the forward computation.

**The key insight: forward + backprop.**  Computing $\pip_i$ requires the "leave-one-out" product $\prod_{j \neq i}(1 + \w_j \z)$—a single item removed from the full product.  A naive approach recomputes this from scratch for each $i$, giving $\mathcal{O}(N^2 n)$ total.  But any differentiable forward pass that computes $\log \Z$—whether the $\mathcal{O}(Nn)$ Pascal DP or the $\mathcal{O}(N \log^2 n)$ product tree—gives all $N$ inclusion probabilities via a single backpropagation pass at the same asymptotic cost.  This is a general principle: backprop computes the gradient of any scalar output for free (up to a small constant factor).  The product tree is just the fastest known forward pass; the gradient follows mechanically.

In the PyTorch implementation, the gradient comes for free via `torch.autograd`—no hand-coded tree traversal needed.

<div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px 20px; margin: 16px 0; width: fit-content; min-width: 100%; max-width: none;">

**Interactive backpropagation tree.** Each node shows the forward polynomial coefficients (<span style="color:#5b9bd5">blue</span>, top) and the adjoint $\bar{c}_k$ (<span style="color:#7b2d8e">purple</span>, bottom).  The seed $\bar{c}_n = 1/Z$ at the root flows upward; at each node, the child's adjoint is the cross-correlation of the parent adjoint with the sibling polynomial.  At the leaves, $\pip_i = \w_i \cdot \bar{c}_1^{(i)}$. Drag the weight sliders to explore.

<div id="bp"></div>
<script>
(function() {
  var N = 4, n = 2;
  var w = [0.5, 1.2, 0.3, 1.0];
  var WMAX = 3;

  var CW = '#5b9bd5', CR = '#c0504d', CI = '#d4a24e', CA = '#7b2d8e'; // blue, red, gold, purple for adjoints
  var root = d3.select('#bp');

  function polyMul(a, b) {
    var out = new Array(a.length + b.length - 1).fill(0);
    for (var i = 0; i < a.length; i++)
      for (var j = 0; j < b.length; j++)
        out[i+j] += a[i] * b[j];
    return out;
  }

  function buildTree(w, n) {
    var leaves = w.map(function(wi, i) {
      return { poly: [1, wi], items: [i], leaf: true };
    });
    var size = 1;
    while (size < leaves.length) size *= 2;
    while (leaves.length < size)
      leaves.push({ poly: [1], items: [], leaf: true, pad: true });
    var level = leaves;
    var allLevels = [level];
    while (level.length > 1) {
      var next = [];
      for (var i = 0; i < level.length; i += 2) {
        var l = level[i], r = level[i+1];
        var p = polyMul(l.poly, r.poly);
        if (p.length > n + 1) p = p.slice(0, n + 1);
        next.push({ left: l, right: r, poly: p, items: l.items.concat(r.items), leaf: false });
      }
      level = next;
      allLevels.push(level);
    }
    return { root: level[0], levels: allLevels };
  }

  // Cross-correlation: adjoint of child through convolution c = conv(child, sibling)
  // childAdj[j] = sum_k parentAdj[j+k] * siblingPoly[k]
  function crossCorr(parentAdj, siblingPoly, maxLen) {
    var out = new Array(maxLen).fill(0);
    for (var j = 0; j < maxLen; j++) {
      for (var k = 0; k < siblingPoly.length; k++) {
        if (j + k < parentAdj.length) {
          out[j] += parentAdj[j + k] * siblingPoly[k];
        }
      }
    }
    return out;
  }

  function backprop(tree, n) {
    var rt = tree.root;
    var Z = rt.poly[Math.min(n, rt.poly.length - 1)] || 0;
    // Seed: d(log Z)/d(c_k) = delta_{k,n} / Z
    var adjLen = Math.min(rt.poly.length, n + 1);
    rt.adj = new Array(adjLen).fill(0);
    if (n < adjLen && Z > 0) rt.adj[n] = 1 / Z;

    (function walkDown(nd) {
      if (nd.leaf || !nd.left) return;
      var left = nd.left, right = nd.right;
      var leftLen = Math.min(left.poly.length, n + 1);
      var rightLen = Math.min(right.poly.length, n + 1);
      left.adj = crossCorr(nd.adj, right.poly, leftLen);
      right.adj = crossCorr(nd.adj, left.poly, rightLen);
      walkDown(left);
      walkDown(right);
    })(rt);

    return Z;
  }

  var mobile = window.innerWidth < 600;
  var bW = mobile ? 22 : 28;
  var bH = mobile ? 50 : 70;
  var halfH = bH;
  var divider = 4;
  var sliderH = bH;
  var bGap = 4;
  var nodePad = 12;
  var vGap = 30;
  var hGap = 20;

  var sliderFills = [], sliderLabels = [];
  var nodeRefs = [];
  var piLabels = [];
  var piBars = [];
  var mathLabels = [];

  function addMathLabel(container, x, y, html, opts) {
    opts = opts || {};
    var div = container.append('div')
      .style('position', 'absolute')
      .style('left', x + 'px')
      .style('top', y + 'px')
      .style('font-size', opts.fontSize || '10px')
      .style('color', opts.color || '#999')
      .style('pointer-events', 'none')
      .style('white-space', 'nowrap');
    if (opts.anchor === 'middle') div.style('transform', 'translateX(-50%)');
    else if (opts.anchor === 'end') div.style('transform', 'translateX(-100%)');
    div.html(html);
    mathLabels.push(div);
    return div;
  }

  function build() {
    root.selectAll('*').remove();
    sliderFills = []; sliderLabels = [];
    nodeRefs = [];
    piLabels = [];
    piBars = [];
    mathLabels = [];

    // Controls
    var ctrl = root.append('div').style('display','flex').style('align-items','center')
      .style('gap','12px').style('margin-bottom','8px').style('font-size','0.9em');
    ctrl.append('span').html('$N$ = ');
    ctrl.append('input').attr('type','number').attr('min',2).attr('value',N)
      .style('width','44px').style('padding','2px 4px').style('font-family','inherit')
      .style('font-size','inherit').style('border','1px solid #ccc').style('border-radius','3px')
      .on('change input', function() {
        var v = Math.max(2, Math.round(+this.value));
        if (v === N) return;
        while (w.length < v) w.push(0.5 + Math.random());
        w = w.slice(0, v); N = v; n = Math.min(n, N); build();
      });
    ctrl.append('span').html('&ensp;$n$ = ');
    ctrl.append('input').attr('type','number').attr('min',0).attr('max',N).attr('value',n)
      .style('width','44px').style('padding','2px 4px').style('font-family','inherit')
      .style('font-size','inherit').style('border','1px solid #ccc').style('border-radius','3px')
      .on('change input', function() {
        var v = Math.max(0, Math.min(N, Math.round(+this.value)));
        if (v === n) return; n = v; build();
      });

    var tree = buildTree(w, n);
    var Z = backprop(tree, n);
    var levels = tree.levels;
    var depth = levels.length;

    // Compute pi values
    var pi = [];
    for (var i = 0; i < N; i++) {
      var leaf = levels[0][i];
      pi.push(leaf.adj && leaf.adj.length > 1 ? w[i] * leaf.adj[1] : 0);
    }

    // Layout
    function nodeBoxW(nd) {
      if (nd.pad) return 0;
      var nc = Math.min(nd.adj ? nd.adj.length : nd.poly.length, n + 1);
      return nc * (bW + bGap) - bGap + nodePad;
    }

    var levelSpacings = levels.map(function(lev) {
      var maxW = 0;
      lev.forEach(function(nd) { var ww = nodeBoxW(nd); if (ww > maxW) maxW = ww; });
      return maxW + hGap;
    });
    levelSpacings[0] = Math.max(levelSpacings[0], bW + hGap + 10);

    var leafCount = levels[0].length;
    var svgW = Math.max(300, leafCount * levelSpacings[0] + 60);

    var labelH = 16;
    var topPad = 10;
    var sepGap = 20;
    var arrowLen = 14;
    var nodeH = halfH * 2 + divider + nodePad;

    // Zone 1: inputs (labels + sliders)
    var inputZoneTop = topPad;
    var inputZoneBot = topPad + labelH + sliderH + 4;
    var sep1Y = inputZoneBot + sepGap/2;
    // Zone 2: tree (leaves at top, root at bottom — same as forward widget)
    var leafY = sep1Y + sepGap/2 + arrowLen;
    var rootBot = leafY + (depth - 1) * (nodeH + vGap) + nodeH;
    var sep2Y = rootBot + sepGap;
    // Zone 3: output (pi bars)
    var piBarH = 50;
    var outputY = sep2Y + sepGap/2 + arrowLen;
    var svgH = outputY + piBarH + 20;

    root.style('width', svgW + 'px');
    var svgWrap = root.append('div')
      .style('position', 'relative')
      .style('width', svgW + 'px')
      .style('height', svgH + 'px');
    var svg = svgWrap.append('svg')
      .attr('width', svgW).attr('height', svgH)
      .style('display', 'block').style('user-select', 'none');

    // Arrow defs
    var defs = svg.append('defs');
    defs.append('marker').attr('id','bpArrowDown').attr('viewBox','0 0 10 10')
      .attr('refX',10).attr('refY',5).attr('markerWidth',7).attr('markerHeight',7).attr('orient','auto')
      .append('path').attr('d','M0,0 L10,5 L0,10 Z').attr('fill','#ccc');
    defs.append('marker').attr('id','bpArrowUp').attr('viewBox','0 0 10 10')
      .attr('refX',0).attr('refY',5).attr('markerWidth',7).attr('markerHeight',7).attr('orient','auto')
      .append('path').attr('d','M10,0 L0,5 L10,10 Z').attr('fill', CA);

    // Assign positions (same as forward widget)
    var leafSpacing = levelSpacings[0];
    var leafStartX = 20;
    for (var ni = 0; ni < leafCount; ni++) {
      levels[0][ni].x = leafStartX + ni * leafSpacing + leafSpacing / 2;
      levels[0][ni].y = leafY;
    }
    for (var li = 1; li < levels.length; li++) {
      levels[li].forEach(function(nd) {
        if (nd.left && nd.right) nd.x = (nd.left.x + nd.right.x) / 2;
        else if (nd.left) nd.x = nd.left.x;
        nd.y = leafY + li * (nodeH + vGap);
      });
    }

    // Separators
    svg.append('line').attr('x1',10).attr('x2',svgW-10).attr('y1',sep1Y).attr('y2',sep1Y)
      .attr('stroke','#e0e0e0').attr('stroke-width',1).attr('stroke-dasharray','4,4');
    svg.append('line').attr('x1',10).attr('x2',svgW-10).attr('y1',sep2Y).attr('y2',sep2Y)
      .attr('stroke','#e0e0e0').attr('stroke-width',1).attr('stroke-dasharray','4,4');

    // Edges (with upward arrows for gradient flow)
    for (var li = 1; li < levels.length; li++) {
      levels[li].forEach(function(nd) {
        [nd.left, nd.right].forEach(function(child) {
          if (!child || child.pad) return;
          var x1 = nd.x, y1 = nd.y;
          var x2 = child.x, y2 = child.y + nodeH;
          var my = (y1 + y2) / 2;
          svg.append('path')
            .attr('d', 'M'+x1+','+y1+' C'+x1+','+my+' '+x2+','+my+' '+x2+','+y2)
            .attr('fill','none').attr('stroke', CA).attr('stroke-width',1.2)
            .attr('stroke-dasharray','4,2').attr('opacity',0.5)
            .attr('marker-end','url(#bpArrowUp)');
        });
      });
    }

    // Draw primal + dual bars at each node (log scale, fixed reference)
    var logMaxFwd = Math.log1p(Math.pow(WMAX, n));
    var logMaxAdj = Math.log1p(1); // adjoints are O(1/Z), start conservative
    for (var li = 0; li < levels.length; li++) {
      levels[li].forEach(function(nd, ni) {
        if (nd.pad) { nodeRefs.push(null); return; }
        var adj = nd.adj || [];
        var coeffs = nd.poly;
        var nCoeffs = Math.min(Math.max(coeffs.length, adj.length), n + 1);
        if (nCoeffs === 0) { nodeRefs.push(null); return; }
        var boxW = nCoeffs * (bW + bGap) - bGap + nodePad;
        var boxH = halfH * 2 + divider + nodePad;
        var gx = nd.x - boxW/2;

        var g = svg.append('g').attr('transform', 'translate(' + gx + ',' + nd.y + ')');
        g.append('rect').attr('width', boxW).attr('height', boxH)
          .attr('fill', '#f9f9f9').attr('stroke', 'none').attr('rx', 5);

        // Divider line between primal and dual
        var divY = nodePad/2 + halfH + divider/2;
        g.append('line').attr('x1', 4).attr('x2', boxW - 4)
          .attr('y1', divY).attr('y2', divY)
          .attr('stroke', '#ddd').attr('stroke-width', 1);

        var fwdRects = [], fwdTexts = [], adjRects = [], adjTexts = [];
        var fwdTop = nodePad/2;
        var adjTop = nodePad/2 + halfH + divider;

        for (var k = 0; k < nCoeffs; k++) {
          var bx = nodePad/2 + k * (bW + bGap);
          var isRoot = (li === levels.length - 1);
          var isHighlight = isRoot && k === n;

          // --- Primal (forward) bar ---
          var fVal = k < coeffs.length ? coeffs[k] : 0;
          var fFrac = Math.min(1, Math.log1p(Math.abs(fVal)) / logMaxFwd);
          var fPx = Math.max(0.5, fFrac * halfH);
          var fCol = isHighlight ? CR : CW;

          g.append('rect').attr('x', bx).attr('y', fwdTop)
            .attr('width', bW).attr('height', halfH)
            .attr('fill', '#f8f8f8').attr('stroke', '#eee').attr('rx', 2);
          var fr = g.append('rect')
            .attr('x', bx + 1).attr('width', bW - 2).attr('rx', 2)
            .attr('y', fwdTop + halfH - fPx).attr('height', fPx)
            .attr('fill', fCol).attr('opacity', 0.7);
          fwdRects.push(fr);
          var ft = g.append('text')
            .attr('x', bx + bW/2).attr('y', fwdTop + halfH - fPx - 3)
            .attr('text-anchor', 'middle')
            .style('font-size', '8px').style('fill', fCol)
            .style('font-family', "'EB Garamond', serif")
            .text(fmtAdj(fVal));
          fwdTexts.push(ft);

          // --- Dual (adjoint) bar ---
          var aVal = k < adj.length ? adj[k] : 0;
          var aFrac = Math.min(1, Math.log1p(Math.abs(aVal)) / logMaxAdj);
          var aPx = Math.max(0.5, aFrac * halfH);
          var isSeed = isRoot && k === n;
          var aCol = isSeed ? CR : CA;

          g.append('rect').attr('x', bx).attr('y', adjTop)
            .attr('width', bW).attr('height', halfH)
            .attr('fill', '#f8f8f8').attr('stroke', '#eee').attr('rx', 2);
          var ar = g.append('rect')
            .attr('x', bx + 1).attr('width', bW - 2).attr('rx', 2)
            .attr('y', adjTop + halfH - aPx).attr('height', aPx)
            .attr('fill', aCol).attr('opacity', 0.7);
          adjRects.push(ar);
          var at2 = g.append('text')
            .attr('x', bx + bW/2).attr('y', adjTop + halfH - aPx - 3)
            .attr('text-anchor', 'middle')
            .style('font-size', '8px').style('fill', aCol)
            .style('font-family', "'EB Garamond', serif")
            .text(fmtAdj(aVal));
          adjTexts.push(at2);
        }

        // Degree labels at root
        if (li === levels.length - 1) {
          for (var k = 0; k < nCoeffs; k++) {
            var labelX = gx + nodePad/2 + k * (bW + bGap) + bW/2;
            addMathLabel(svgWrap, labelX, nd.y + boxH + 2,
              k === n ? '$\\mathbf{' + k + '}$' : '$' + k + '$',
              {anchor:'middle', color: k === n ? CR : '#bbb', fontSize:'9px'});
          }
        }

        nd._barXs = [];
        for (var k = 0; k < nCoeffs; k++) {
          nd._barXs.push(gx + nodePad/2 + k * (bW + bGap));
        }

        nodeRefs.push({ fwdRects: fwdRects, fwdTexts: fwdTexts, adjRects: adjRects, adjTexts: adjTexts, nCoeffs: nCoeffs });
      });
    }

    // Connector arrows from slider bottom to leaf
    levels[0].forEach(function(nd, idx) {
      if (nd.pad || !nd._barXs) return;
      var barIdx = Math.min(1, nd._barXs.length - 1);
      var cx = nd._barXs[barIdx] + bW/2;
      svg.append('line')
        .attr('x1', cx).attr('y1', inputZoneBot)
        .attr('x2', cx).attr('y2', nd.y)
        .attr('stroke', '#ccc').attr('stroke-width', 1)
        .attr('marker-end', 'url(#bpArrowDown)');
    });

    // Sliders above leaves
    levels[0].forEach(function(nd, idx) {
      if (nd.pad) return;
      (function(idx) {
        var barIdx = Math.min(1, nd._barXs.length - 1);
        var barX = nd._barXs[barIdx];
        var sliderTop = topPad + labelH;

        var sg = svg.append('g');
        sg.append('rect').attr('x', barX + 1).attr('y', sliderTop)
          .attr('width', bW - 2).attr('height', sliderH)
          .attr('fill', '#f8f8f8').attr('stroke', '#eee').attr('rx', 1);

        var frac = Math.min(w[idx] / WMAX, 1);
        var sf = sg.append('rect')
          .attr('x', barX + 2).attr('width', bW - 4).attr('rx', 1)
          .attr('y', sliderTop + sliderH - frac * sliderH).attr('height', frac * sliderH)
          .attr('fill', CI).attr('opacity', 0.8).style('pointer-events', 'none');
        sliderFills.push(sf);

        var sl = sg.append('text')
          .attr('x', barX + bW/2).attr('y', sliderTop + sliderH - frac * sliderH - 2)
          .attr('text-anchor', 'middle')
          .style('font-size', '9px').style('fill', CI).style('pointer-events', 'none')
          .text(w[idx].toFixed(2));
        sliderLabels.push(sl);

        addMathLabel(svgWrap, barX + bW/2, topPad, '$w_{' + (idx+1) + '}$', {anchor:'middle', color:CI, fontSize:'11px'});

        sg.append('rect')
          .attr('x', barX - 4).attr('y', sliderTop - 2)
          .attr('width', bW + 8).attr('height', sliderH + 4)
          .attr('fill', 'transparent').attr('cursor', 'ns-resize')
          .style('touch-action', 'none')
          .call(d3.drag().on('drag', function(event) {
            var fr = (sliderTop + sliderH - event.y) / sliderH;
            w[idx] = Math.max(0.01, Math.min(WMAX, fr * WMAX));
            updateBP();
          }));
      })(idx);
    });

    // Output zone: pi_i bars
    var piBarW = bW;
    levels[0].forEach(function(nd, idx) {
      if (nd.pad || !nd._barXs) return;
      var barIdx = Math.min(1, nd._barXs.length - 1);
      var bx = nd._barXs[barIdx];

      // Track
      svg.append('rect').attr('x', bx + 1).attr('y', outputY)
        .attr('width', piBarW - 2).attr('height', piBarH)
        .attr('fill', '#f8f8f8').attr('stroke', '#eee').attr('rx', 2);

      var piFrac = Math.min(pi[idx], 1);
      var pb = svg.append('rect')
        .datum({ oy: outputY, h: piBarH })
        .attr('x', bx + 2).attr('width', piBarW - 4).attr('rx', 2)
        .attr('y', outputY + piBarH - piFrac * piBarH).attr('height', piFrac * piBarH)
        .attr('fill', CR).attr('opacity', 0.7);
      piBars.push(pb);

      var pl = svg.append('text')
        .attr('x', bx + piBarW/2).attr('y', outputY + piBarH - piFrac * piBarH - 3)
        .attr('text-anchor', 'middle')
        .style('font-size', '9px').style('fill', CR)
        .style('font-family', "'EB Garamond', serif")
        .text(pi[idx].toFixed(3));
      piLabels.push(pl);

      // Arrow from leaf adjoint bar to pi output
      var leafBot = nd.y + nodeH + 8;
      svg.append('line')
        .attr('x1', bx + piBarW/2).attr('y1', leafBot)
        .attr('x2', bx + piBarW/2).attr('y2', outputY)
        .attr('stroke', '#d4a0a0').attr('stroke-width', 1)
        .attr('marker-end', 'url(#bpArrowDown)');

      addMathLabel(svgWrap, bx + piBarW/2, outputY + piBarH + 3,
        '$\\pi_{' + (idx+1) + '}$', {anchor:'middle', color:CR, fontSize:'11px'});
    });

    // Seed annotation at root
    var rootNd = levels[levels.length - 1][0];
    if (rootNd._barXs && n < rootNd._barXs.length) {
      addMathLabel(svgWrap, rootNd._barXs[n] + bW/2, rootNd.y - 14,
        '$\\bar{c}_n = 1/Z$', {anchor:'middle', color:CR, fontSize:'10px'});
    }

    root.datum({ levels: levels, tree: tree });
    updateBP();

    setTimeout(function() {
      if (window.MathJax && MathJax.typesetPromise) {
        var el = document.getElementById('bp');
        MathJax.typesetClear([el]);
        MathJax.typesetPromise([el]);
      }
    }, 10);
  }

  function fmtAdj(v) {
    if (v === 0) return '0';
    if (Math.abs(v) < 0.005) return '';
    if (Math.abs(v) >= 10) return v.toFixed(0);
    if (Math.abs(v) >= 1) return v.toFixed(1);
    return v.toFixed(2);
  }

  function updateBP() {
    var sliderTop = 10 + 16;

    // Update sliders
    for (var i = 0; i < N; i++) {
      var frac = Math.min(w[i] / WMAX, 1);
      sliderFills[i].attr('y', sliderTop + sliderH - frac * sliderH).attr('height', frac * sliderH);
      sliderLabels[i].attr('y', sliderTop + sliderH - frac * sliderH - 2).text(w[i].toFixed(2));
    }

    // Rebuild tree + backprop
    var newTree = buildTree(w, n);
    var Z = backprop(newTree, n);
    var newLevels = newTree.levels;

    var logMaxFwd = Math.log1p(Math.pow(WMAX, n));
    var logMaxAdj = Math.log1p(1);
    var fwdTop = nodePad/2;
    var adjTop = nodePad/2 + halfH + divider;

    // Update node bars (log scale, fixed reference)
    var ri = 0;
    for (var li = 0; li < newLevels.length; li++) {
      for (var ni = 0; ni < newLevels[li].length; ni++) {
        var ref = nodeRefs[ri++];
        if (!ref) continue;
        var nd = newLevels[li][ni];
        var coeffs = nd.poly;
        var adj = nd.adj || [];
        var nCoeffs = ref.nCoeffs;
        for (var k = 0; k < nCoeffs; k++) {
          // Primal
          var fVal = k < coeffs.length ? coeffs[k] : 0;
          var fFrac = Math.min(1, Math.log1p(Math.abs(fVal)) / logMaxFwd);
          var fPx = Math.max(1, fFrac * halfH);
          ref.fwdRects[k].attr('y', fwdTop + halfH - fPx).attr('height', fPx);
          ref.fwdTexts[k].attr('y', fwdTop + halfH - fPx - 3).text(fmtAdj(fVal));

          // Dual
          var aVal = k < adj.length ? adj[k] : 0;
          var aFrac = Math.min(1, Math.log1p(Math.abs(aVal)) / logMaxAdj);
          var aPx = Math.max(1, aFrac * halfH);
          ref.adjRects[k].attr('y', adjTop + halfH - aPx).attr('height', aPx);
          ref.adjTexts[k].attr('y', adjTop + halfH - aPx - 3).text(fmtAdj(aVal));
        }
      }
    }

    // Update pi bars
    var piArr = [];
    for (var i = 0; i < N; i++) {
      var leaf = newLevels[0][i];
      piArr.push(leaf.adj && leaf.adj.length > 1 ? w[i] * leaf.adj[1] : 0);
    }
    for (var i = 0; i < piArr.length; i++) {
      if (i < piBars.length) {
        var d = piBars[i].datum();
        var piFrac = Math.min(piArr[i], 1);
        piBars[i].attr('y', d.oy + d.h - piFrac * d.h).attr('height', piFrac * d.h);
        piLabels[i].attr('y', d.oy + d.h - piFrac * d.h - 3).text(piArr[i].toFixed(3));
      }
    }
  }

  build();
})();
</script>

</div>

### Drawing Exact Samples

Sampling reuses the product tree (no gradient computation needed).  Starting at the root with a quota of $k = n$ items to select, we walk top-down: at each internal node, randomly split the quota between the left and right subtrees.  The probability of assigning $j$ items to the left is proportional to $\llbracket P_L \rrbracket(\z^j) \cdot \llbracket P_R \rrbracket(\z^{k-j})$.  This is exact, not approximate—it follows from the **weighted Vandermonde identity**: $\Zw{(\ba;\, \bb)}{k} = \sum_{j=0}^{k} \Zw{\ba}{j} \cdot \Zw{\bb}{k-j}$, which says the number of ways to choose $k$ items from $A \cup B$ is the convolution of the two groups' counts.  Each term in the sum is the conditional probability of a particular left/right split.  At the leaves, quota 1 means "include this item"; quota 0 means "exclude."

<style>
#sampling-anim .sa-widget-box {
  background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 16px 20px;
  margin: 16px 0;
}
#sampling-anim .sa-widget-box p.desc { margin-bottom: 12px; font-size: 0.95em; line-height: 1.5; }
#sampling-anim #sa-controls {
  display: flex; align-items: center; gap: 8px;
  margin-bottom: 8px; font-size: 0.9em;
  flex-wrap: wrap;
}
#sampling-anim #sa-controls input[type=number] {
  width: 44px; padding: 2px 4px;
  font-family: inherit; font-size: inherit;
  border: 1px solid #ccc; border-radius: 3px;
}
#sampling-anim #sa-controls button {
  font-family: inherit; font-size: 0.85em;
  border: 1px solid #ccc; border-radius: 3px;
  padding: 3px 10px; cursor: pointer; color: #555;
  background: #fff;
}
#sampling-anim #sa-controls button:hover { background: #f0f0f0; }
#sampling-anim #sa-controls button:disabled { opacity: 0.35; cursor: default; }
#sampling-anim #sa-controls button:disabled:hover { background: #fff; }
#sampling-anim #sa-controls button.active { background: #5b9bd5; color: #fff; border-color: #5b9bd5; }
#sampling-anim #sa-status-bar {
  font-size: 0.82em; color: #666; margin-top: 4px;
  min-height: 1.4em; line-height: 1.5;
}
#sampling-anim #sa-result-bar {
  margin-top: 6px; font-size: 1em; min-height: 1.6em;
  padding: 4px 0;
}
#sampling-anim svg text { font-family: 'EB Garamond', 'Georgia', serif; }
#sampling-anim .leaf-icon, #sampling-anim .quota-ball, #sampling-anim .anim-ball,
#sampling-anim .quota-num, #sampling-anim .hover-overlay, #sampling-anim .split-overlay {
  pointer-events: none;
}
</style>

<div id="sampling-anim">
<div class="sa-widget-box">
<p class="desc">
<strong>Sampling animation.</strong>
Starting at the root with a quota of $n$ items, the algorithm walks the product tree top-down.
At each internal node with quota $k$, it splits $k$ between left and right children with probability
proportional to $P_L[j] \cdot P_R[k{-}j]$&mdash;the <strong>weighted Vandermonde identity</strong>.
At the leaves, quota 1 = selected, quota 0 = excluded.
Drag the weight sliders to change $w_i$ and see how the split distributions update.
</p>

<div id="sa-controls">
  <span>$N$ = </span><input type="number" id="sa-inp-N" min="2" max="16" value="8">
  <span>&ensp;$n$ = </span><input type="number" id="sa-inp-n" min="0" max="8" value="3">
  <button id="sa-btn-back" disabled>&laquo; Back</button>
  <button id="sa-btn-step">Step &raquo;</button>
  <button id="sa-btn-play">Play</button>
  <button id="sa-btn-weights">Resample Weights</button>
</div>

<div id="sa-anim-container" style="position: relative; overflow: visible;"></div>
<div id="sa-status-bar"></div>
<div id="sa-result-bar"></div>
</div>
</div>

<script>
(function() {
  // ── Colors (matching blog widgets) ──
  var CW = '#5b9bd5', CR = '#c0504d', CI = '#d4a24e';
  var CS = '#4caf50';   // selected green
  var CX = '#ccc';      // excluded gray

  var N = 8, n = 3;
  var w = [];
  var dur = 500;
  var pauseDur = 600;
  var tweenDur = 250;  // transition duration for slider-driven changes

  // ── Layout ──
  var ballR = 6;
  var nodeW = 44, nodeH = 28;
  var pmfBarW = 6, pmfBarH = 20, pmfGap = 2;
  var vGap = 90;
  var sliderH = 50, sliderW, maxW = 3.0;
  var sliderFills = [], sliderLabels = [];

  // ── State ──
  var tree, flatNodes, treeDepth;
  var animState = null;
  var uniforms = {};      // pre-drawn U[0,1] per node id — common random numbers
  var history = [];       // stack of snapshots for back-stepping
  var prevRun = null;     // saved state from previous run (for undo past reset)
  var animating = false;  // true while ball animation is in flight
  var playing = false;
  var playTimer = null;

  // ── Math ──
  function polyMul(a, b) {
    var out = new Array(a.length + b.length - 1).fill(0);
    for (var i = 0; i < a.length; i++)
      for (var j = 0; j < b.length; j++)
        out[i+j] += a[i] * b[j];
    return out;
  }

  function buildTree(ww, nn) {
    var leaves = ww.map(function(wi, i) {
      return { poly: [1, wi], items: [i], leaf: true, id: 'leaf-' + i };
    });
    var size = 1;
    while (size < leaves.length) size *= 2;
    while (leaves.length < size)
      leaves.push({ poly: [1], items: [], leaf: true, pad: true, id: 'pad-' + leaves.length });
    var level = leaves;
    var allLevels = [level];
    var nid = 0;
    while (level.length > 1) {
      var next = [];
      for (var i = 0; i < level.length; i += 2) {
        var l = level[i], r = level[i+1];
        var p = polyMul(l.poly, r.poly);
        if (p.length > nn + 1) p = p.slice(0, nn + 1);
        next.push({ left: l, right: r, poly: p, items: l.items.concat(r.items), leaf: false, id: 'int-' + (nid++) });
      }
      level = next;
      allLevels.push(level);
    }
    allLevels.reverse();
    return { root: allLevels[0][0], levels: allLevels };
  }

  function randomWeights(NN) {
    var ww = [];
    for (var i = 0; i < NN; i++) ww.push(0.3 + Math.random() * 2.2);
    return ww;
  }

  function initWeights() { w = randomWeights(N); }

  function splitDist(nodeLeft, nodeRight, k) {
    var probs = [], total = 0;
    for (var j = 0; j <= k; j++) {
      var pl = j < nodeLeft.poly.length ? nodeLeft.poly[j] : 0;
      var pr = (k - j) < nodeRight.poly.length ? nodeRight.poly[k - j] : 0;
      var v = Math.max(pl * pr, 0);
      probs.push(v);
      total += v;
    }
    if (total > 0) for (var j = 0; j < probs.length; j++) probs[j] /= total;
    return probs;
  }

  function sampleWithU(probs, u) {
    var cum = 0;
    for (var j = 0; j < probs.length; j++) {
      cum += probs[j];
      if (u < cum) return j;
    }
    return probs.length - 1;
  }

  function ensureUniforms() {
    flatNodes.forEach(function(nd) {
      if (!nd.leaf && !nd.pad && uniforms[nd.id] === undefined) {
        uniforms[nd.id] = Math.random();
      }
    });
  }

  function polyToPMF(poly) {
    var total = 0;
    for (var i = 0; i < poly.length; i++) total += poly[i];
    if (total === 0) return poly.slice();
    return poly.map(function(v) { return v / total; });
  }

  // ── Snapshots for back-stepping ──
  function snapshot() {
    return {
      level: animState.level,
      quotas: Object.assign({}, animState.quotas),
      splits: Object.assign({}, animState.splits),
      done: animState.done
    };
  }

  function restoreSnapshot(snap) {
    animState.level = snap.level;
    animState.quotas = Object.assign({}, snap.quotas);
    animState.splits = Object.assign({}, snap.splits);
    animState.done = snap.done;
  }

  // ── Layout ──
  function layoutTree() {
    var levels = tree.levels;
    treeDepth = levels.length;
    flatNodes = [];
    for (var li = 0; li < levels.length; li++) {
      for (var ni = 0; ni < levels[li].length; ni++) {
        var nd = levels[li][ni];
        nd._level = li;
        nd._idx = ni;
        if (!nd.pad) flatNodes.push(nd);
      }
    }
    var leafLevel = levels[levels.length - 1];
    var leafCount = leafLevel.filter(function(nd) { return !nd.pad; }).length;
    var hSpacing = nodeW + 14;
    var totalW = Math.max(leafCount * hSpacing, 280);
    var realIdx = 0;
    for (var i = 0; i < leafLevel.length; i++) {
      var nd = leafLevel[i];
      if (nd.pad) { nd.x = -9999; nd.y = -9999; continue; }
      nd.x = 20 + realIdx * hSpacing + hSpacing / 2;
      nd.y = (treeDepth - 1) * vGap + 36;
      realIdx++;
    }
    for (var li = levels.length - 2; li >= 0; li--) {
      for (var ni = 0; ni < levels[li].length; ni++) {
        var nd = levels[li][ni];
        var lp = nd.left && !nd.left.pad;
        var rp = nd.right && !nd.right.pad;
        if (lp && rp) {
          nd.x = (nd.left.x + nd.right.x) / 2;
        } else if (lp) {
          nd.x = nd.left.x;
        } else if (rp) {
          nd.x = nd.right.x;
        } else {
          nd.pad = true;
          nd.x = -9999; nd.y = -9999;
          continue;
        }
        nd.y = li * vGap + 36;
      }
    }
    flatNodes = [];
    for (var li = 0; li < levels.length; li++) {
      for (var ni = 0; ni < levels[li].length; ni++) {
        var nd = levels[li][ni];
        nd._level = li;
        nd._idx = ni;
        if (!nd.pad) flatNodes.push(nd);
      }
    }
    return { svgW: totalW + 40, svgH: treeDepth * vGap + 46 + sliderH + 28 };
  }

  // ── Drawing ──
  var svg, container;

  function draw() {
    container = d3.select('#sa-anim-container');
    drawScaffolding();
    resetAnimState();
    renderState();
  }

  function drawPMF(g, nd) {
    var pmf = polyToPMF(nd.poly);
    var maxP = Math.max.apply(null, pmf);
    if (maxP === 0) maxP = 1;
    var nBars = pmf.length;
    var totalBarW = nBars * (pmfBarW + pmfGap) - pmfGap;
    var startX = (nodeW - totalBarW) / 2;
    var topY = nodeH + 4;

    var pmfG = g.append('g').attr('class', 'pmf-hist');

    for (var k = 0; k < nBars; k++) {
      var h = Math.max(0.5, pmf[k] / maxP * pmfBarH);
      pmfG.append('rect')
        .attr('x', startX + k * (pmfBarW + pmfGap))
        .attr('y', topY + pmfBarH - h)
        .attr('width', pmfBarW)
        .attr('height', h)
        .attr('rx', 1)
        .attr('fill', CW)
        .attr('opacity', 0.25);
    }
  }

  function subscript(nn) {
    var digits = '₀₁₂₃₄₅₆₇₈₉';
    return String(nn).split('').map(function(d) { return digits[+d]; }).join('');
  }

  function buildSliders() {
    sliderFills = []; sliderLabels = [];
    sliderW = nodeW - 10;

    var leafLevel = tree.levels[tree.levels.length - 1];
    leafLevel.forEach(function(nd) {
      if (nd.pad || !nd.items.length) return;
      var idx = nd.items[0];
      (function(idx) {
        var sx = nd.x - sliderW / 2;
        var sy = nd.y + nodeH / 2 + pmfBarH + 10;

        var sg = svg.append('g').attr('class', 'weight-slider');

        sg.append('text')
          .attr('x', nd.x).attr('y', sy + sliderH + 12)
          .attr('text-anchor', 'middle')
          .style('font-size', '13px').style('fill', CI)
          .style('font-family', "'EB Garamond', serif")
          .text('w' + subscript(idx + 1));

        sg.append('rect')
          .attr('x', sx + 1).attr('y', sy)
          .attr('width', sliderW - 2).attr('height', sliderH)
          .attr('fill', '#f8f8f8').attr('stroke', '#eee').attr('rx', 1);

        var frac = Math.min(w[idx] / maxW, 1);
        var sf = sg.append('rect')
          .attr('x', sx + 2).attr('width', sliderW - 4).attr('rx', 1)
          .attr('y', sy + sliderH - frac * sliderH).attr('height', frac * sliderH)
          .attr('fill', CI).attr('opacity', 0.8)
          .style('pointer-events', 'none');
        sliderFills.push(sf);

        var sl = sg.append('text')
          .attr('x', nd.x).attr('y', sy + sliderH - frac * sliderH - 2)
          .attr('text-anchor', 'middle')
          .style('font-size', '9px').style('fill', CI)
          .style('pointer-events', 'none')
          .text(w[idx].toFixed(2));
        sliderLabels.push(sl);

        sg.append('rect')
          .attr('x', sx - 4).attr('y', sy - 2)
          .attr('width', sliderW + 8).attr('height', sliderH + 4)
          .attr('fill', 'transparent').attr('cursor', 'ns-resize')
          .style('touch-action', 'none')
          .call(d3.drag().on('drag', function(event) {
            var frac = (sy + sliderH - event.y) / sliderH;
            frac = Math.max(0.02, Math.min(1.0, frac));
            w[idx] = frac * maxW;
            sf.attr('y', sy + sliderH - frac * sliderH).attr('height', frac * sliderH);
            sl.attr('y', sy + sliderH - frac * sliderH - 2).text(w[idx].toFixed(2));
            onWeightDrag();
          }));
      })(idx);
    });
  }

  function syncSliders() {
    for (var i = 0; i < sliderFills.length && i < N; i++) {
      var leafLevel = tree.levels[tree.levels.length - 1];
      var nd = leafLevel[i];
      if (!nd || nd.pad) continue;
      var sy = nd.y + nodeH / 2 + pmfBarH + 10;
      var frac = Math.min(w[i] / maxW, 1);
      sliderFills[i].attr('y', sy + sliderH - frac * sliderH).attr('height', frac * sliderH);
      sliderLabels[i].attr('y', sy + sliderH - frac * sliderH - 2).text(w[i].toFixed(2));
    }
  }

  function replaySplits(targetIdx, wasDone) {
    animState = { level: -1, quotas: {}, splits: {}, done: false };
    animState.quotas[tree.root.id] = n;
    nodeQueue = buildNodeQueue();
    queueIdx = 0;
    history = [];

    var target = wasDone ? nodeQueue.length : targetIdx;
    while (queueIdx < target && queueIdx < nodeQueue.length) {
      var nd = nodeQueue[queueIdx];
      var q = animState.quotas[nd.id];
      if (q === undefined || q === 0) {
        if (nd.left && !nd.left.pad) animState.quotas[nd.left.id] = 0;
        if (nd.right && !nd.right.pad) animState.quotas[nd.right.id] = 0;
        queueIdx++;
        continue;
      }
      history.push(snapshot());
      var k = q;
      var probs = splitDist(nd.left, nd.right, k);
      var u = uniforms[nd.id];
      var j = sampleWithU(probs, u);
      animState.splits[nd.id] = { probs: probs, j: j, k: k, u: u };
      animState.quotas[nd.left.id] = j;
      animState.quotas[nd.right.id] = k - j;
      animState.level = nd._level;
      queueIdx++;
    }

    var allDone = true;
    for (var qi = queueIdx; qi < nodeQueue.length; qi++) {
      var rq = animState.quotas[nodeQueue[qi].id];
      if (rq !== undefined && rq > 0) { allDone = false; break; }
    }
    if (allDone || wasDone) {
      for (var qi = queueIdx; qi < nodeQueue.length; qi++) {
        var remaining = nodeQueue[qi];
        if (remaining.left && !remaining.left.pad) animState.quotas[remaining.left.id] = 0;
        if (remaining.right && !remaining.right.pad) animState.quotas[remaining.right.id] = 0;
      }
      queueIdx = nodeQueue.length;
      animState.done = true;
    }
  }

  function onWeightDrag() {
    stopPlaying();
    var prevQueueIdx = queueIdx;
    var wasDone = animState ? animState.done : false;

    tree = buildTree(w, n);
    layoutTree();
    ensureUniforms();

    replaySplits(prevQueueIdx, wasDone);

    updatePMFs();
    renderStateAnimated();
    updateStatusBar();
    showCurrentOverlay();
  }

  function onWeightsChanged() {
    stopPlaying();
    drawScaffolding();
    ensureUniforms();
    replaySplits(0, false);
    renderState();
    updateStatusBar();
    showCurrentOverlay();
  }

  function updatePMFs() {
    flatNodes.forEach(function(nd) {
      var g = svg.select('[data-id="'+nd.id+'"]');
      if (g.empty()) return;
      var pmfG = g.select('.pmf-hist');
      if (pmfG.empty()) return;

      var pmf = polyToPMF(nd.poly);
      var maxP = Math.max.apply(null, pmf);
      if (maxP === 0) maxP = 1;
      var topY = nodeH + 4;

      var q = animState.quotas[nd.id];
      var visited = isNodeVisited(nd);
      var highlightIdx = -1;
      if (visited && q !== undefined && q >= 0 && q < pmf.length) {
        if (nd.leaf) {
          if (q > 0) highlightIdx = q;
        } else {
          highlightIdx = q;
        }
      }

      pmfG.selectAll('rect').each(function(d, i) {
        if (i < pmf.length) {
          var h = Math.max(0.5, pmf[i] / maxP * pmfBarH);
          var isHigh = (i === highlightIdx);
          d3.select(this)
            .transition().duration(tweenDur)
            .attr('y', topY + pmfBarH - h)
            .attr('height', h)
            .attr('fill', isHigh ? CR : CW)
            .attr('opacity', isHigh ? 0.85 : 0.25);
        }
      });
    });
  }

  function renderStateAnimated() {
    var t = d3.transition().duration(tweenDur);

    flatNodes.forEach(function(nd) {
      var g = svg.select('[data-id="'+nd.id+'"]');
      var q = animState.quotas[nd.id];
      var isActive = (q !== undefined && q > 0);
      var isZero = (q !== undefined && q === 0);
      var isProcessed = animState.splits[nd.id] !== undefined;

      var rect = g.select('.node-box');
      if (isActive) {
        rect.transition(t).attr('fill', '#e8f0fa').attr('stroke', CW).attr('stroke-width', 1.5);
      } else if (isZero && nd.leaf) {
        rect.transition(t).attr('fill', '#fafafa').attr('stroke', '#ddd').attr('stroke-width', 1);
      } else if (isProcessed) {
        rect.transition(t).attr('fill', '#f5f5f5').attr('stroke', '#d0d0d0').attr('stroke-width', 1);
      } else {
        rect.transition(t).attr('fill', '#f9f9f9').attr('stroke', '#e0e0e0').attr('stroke-width', 1);
      }
    });

    svg.selectAll('.quota-ball').remove();
    svg.selectAll('.quota-num').remove();
    svg.selectAll('.leaf-icon').remove();

    flatNodes.forEach(function(nd) {
      var q = animState.quotas[nd.id];
      if (q === undefined) return;

      var decided = isNodeDecided(nd);

      if (nd.leaf && nd.items.length && decided) {
        var g = svg.select('[data-id="'+nd.id+'"]');
        var rect = g.select('.node-box');
        if (q === 1) {
          rect.transition(t).attr('fill', '#e8f5e9').attr('stroke', CS).attr('stroke-width', 2);
          g.append('text')
            .attr('class', 'leaf-icon leaf-status')
            .attr('x', nodeW / 2).attr('y', nodeH / 2 + 4)
            .attr('text-anchor', 'middle')
            .style('font-size', '14px').style('fill', CS).style('font-weight', 'bold')
            .text('\u2713');
        } else {
          rect.transition(t).attr('fill', '#fafafa').attr('stroke', '#ddd').attr('stroke-width', 1);
          g.append('text')
            .attr('class', 'leaf-icon leaf-status')
            .attr('x', nodeW / 2).attr('y', nodeH / 2 + 4)
            .attr('text-anchor', 'middle')
            .style('font-size', '12px').style('fill', '#ccc')
            .text('\u2717');
        }
        return;
      }

      if (q === 0 && decided) {
        var g = svg.select('[data-id="'+nd.id+'"]');
        var rect = g.select('.node-box');
        rect.transition(t).attr('fill', '#fafafa').attr('stroke', '#ddd').attr('stroke-width', 1);
        g.append('text')
          .attr('class', 'leaf-icon')
          .attr('x', nodeW / 2).attr('y', nodeH / 2 + 4)
          .attr('text-anchor', 'middle')
          .style('font-size', '12px').style('fill', '#ccc')
          .text('\u2717');
        return;
      }

      if (q <= 0) return;
      drawStaticBalls(nd, q);
    });

    svg.selectAll('.tree-edge').each(function() {
      var edge = d3.select(this);
      var pid = edge.attr('data-parent');
      var cid = edge.attr('data-child');
      var parentSplit = animState.splits[pid];
      var childQ = animState.quotas[cid];

      if (parentSplit && childQ !== undefined && childQ > 0) {
        edge.transition(t).attr('stroke', CW).attr('stroke-width', 2.5).attr('opacity', 0.8);
      } else if (parentSplit && childQ === 0) {
        edge.transition(t).attr('stroke', '#ddd').attr('stroke-width', 1).attr('opacity', 0.4);
      } else if (parentSplit) {
        edge.transition(t).attr('stroke', '#ccc').attr('stroke-width', 1.2).attr('opacity', 0.5);
      } else {
        edge.transition(t).attr('stroke', '#ddd').attr('stroke-width', 1.5).attr('opacity', 1);
      }
    });

    if (animState.done) {
      var selected = [];
      flatNodes.forEach(function(nd) {
        if (nd.leaf && nd.items.length === 1 && animState.quotas[nd.id] === 1)
          selected.push(nd.items[0] + 1);
      });
      d3.select('#sa-result-bar').html(
        '<span style="color:'+CS+'; font-weight:bold;">Sample:</span> $S = \\{' + selected.join(', ') + '\\}$'
      );
      typesetMath();
    } else {
      d3.select('#sa-result-bar').html('');
    }

    updateButtons();
  }

  function drawScaffolding() {
    container.selectAll('*').remove();
    overlayNodeId = null; overlayIsPreview = null; overlayK = -1;
    tree = buildTree(w, n);
    var dims = layoutTree();

    svg = container.append('svg')
      .attr('width', dims.svgW)
      .attr('height', dims.svgH)
      .style('display', 'block')
      .style('user-select', 'none')
      .style('overflow', 'visible');

    var levels = tree.levels;
    for (var li = 0; li < levels.length - 1; li++) {
      levels[li].forEach(function(nd) {
        [nd.left, nd.right].forEach(function(child) {
          if (!child || child.pad) return;
          var x1 = nd.x, y1 = nd.y + nodeH / 2;
          var x2 = child.x, y2 = child.y - nodeH / 2;
          var my = (y1 + y2) / 2;
          svg.append('path')
            .attr('class', 'tree-edge')
            .attr('data-parent', nd.id)
            .attr('data-child', child.id)
            .attr('d', 'M'+x1+','+y1+' C'+x1+','+my+' '+x2+','+my+' '+x2+','+y2)
            .attr('fill', 'none')
            .attr('stroke', '#ddd')
            .attr('stroke-width', 1.5);
        });
      });
    }
    flatNodes.forEach(function(nd) {
      var g = svg.append('g')
        .attr('class', 'tree-node')
        .attr('data-id', nd.id)
        .attr('transform', 'translate('+(nd.x - nodeW/2)+','+(nd.y - nodeH/2)+')');
      g.append('rect')
        .attr('class', 'node-box')
        .attr('width', nodeW).attr('height', nodeH)
        .attr('rx', 4)
        .attr('fill', '#f9f9f9').attr('stroke', '#e0e0e0').attr('stroke-width', 1);
      if (!nd.pad) drawPMF(g, nd);

      if (!nd.leaf && !nd.pad) {
        g.style('cursor', 'pointer')
          .on('mouseenter', (function(nd) { return function() { showHoverOverlay(nd); }; })(nd))
          .on('mouseleave', function() { hideHoverOverlay(); });
      }
    });
    buildSliders();
  }

  function resetAnimState(keepUniforms) {
    if (animState && (history.length > 0 || animState.done)) {
      prevRun = {
        animState: snapshot(),
        history: history.slice(),
        uniforms: Object.assign({}, uniforms),
        queueIdx: queueIdx
      };
    }
    animState = {
      level: -1,
      quotas: {},
      splits: {},
      done: false
    };
    animState.quotas[tree.root.id] = n;
    nodeQueue = buildNodeQueue();
    queueIdx = 0;
    history = [];
    animating = false;
    if (!keepUniforms) {
      uniforms = {};
      ensureUniforms();
    }
    updateStatusBar();
    d3.select('#sa-result-bar').html('');
    updateButtons();
    showCurrentOverlay();
  }

  function updateStatusBar() {
    if (animState.done) {
      d3.select('#sa-status-bar').html('<b>Done!</b> All leaves assigned.');
    } else {
      var nextNd = findNextActiveNode();
      if (nextNd) {
        var items = '{' + nextNd.items.map(function(i){return i+1;}).join(',') + '}';
        var k = animState.quotas[nextNd.id];
        d3.select('#sa-status-bar').html(
          '<b>Level ' + nextNd._level + '</b> \u2014 ' +
          items + ': split quota ' + k + '. Press <b>Step</b>.'
        );
      } else {
        d3.select('#sa-status-bar').html('Press <b>Step</b> to advance.');
      }
    }
  }

  function updateButtons() {
    d3.select('#sa-btn-back').property('disabled', (history.length === 0 && !prevRun) || animating);

    var stepBtn = d3.select('#sa-btn-step');
    if (animState.done) {
      stepBtn.text('Reset').property('disabled', false);
    } else {
      stepBtn.text('Step \u00bb').property('disabled', animating);
    }
  }

  function renderState() {
    flatNodes.forEach(function(nd) {
      var g = svg.select('[data-id="'+nd.id+'"]');
      var q = animState.quotas[nd.id];
      var isActive = (q !== undefined && q > 0);
      var isZero = (q !== undefined && q === 0);
      var isProcessed = animState.splits[nd.id] !== undefined;

      var rect = g.select('.node-box');
      if (isActive) {
        rect.attr('fill', '#e8f0fa').attr('stroke', CW).attr('stroke-width', 1.5);
      } else if (isZero && nd.leaf) {
        rect.attr('fill', '#fafafa').attr('stroke', '#ddd').attr('stroke-width', 1);
      } else if (isProcessed) {
        rect.attr('fill', '#f5f5f5').attr('stroke', '#d0d0d0').attr('stroke-width', 1);
      } else {
        rect.attr('fill', '#f9f9f9').attr('stroke', '#e0e0e0').attr('stroke-width', 1);
      }

      highlightPMFBar(g, nd, q);
    });

    svg.selectAll('.quota-ball').remove();
    svg.selectAll('.quota-num').remove();
    svg.selectAll('.leaf-icon').remove();

    flatNodes.forEach(function(nd) {
      var q = animState.quotas[nd.id];
      if (q === undefined) return;

      var decided = isNodeDecided(nd);

      if (nd.leaf && nd.items.length && decided) {
        var g = svg.select('[data-id="'+nd.id+'"]');
        var rect = g.select('.node-box');
        if (q === 1) {
          rect.attr('fill', '#e8f5e9').attr('stroke', CS).attr('stroke-width', 2);
          g.append('text')
            .attr('class', 'leaf-icon leaf-status')
            .attr('x', nodeW / 2).attr('y', nodeH / 2 + 4)
            .attr('text-anchor', 'middle')
            .style('font-size', '14px').style('fill', CS).style('font-weight', 'bold')
            .text('\u2713');
        } else {
          rect.attr('fill', '#fafafa').attr('stroke', '#ddd').attr('stroke-width', 1);
          g.append('text')
            .attr('class', 'leaf-icon leaf-status')
            .attr('x', nodeW / 2).attr('y', nodeH / 2 + 4)
            .attr('text-anchor', 'middle')
            .style('font-size', '12px').style('fill', '#ccc')
            .text('\u2717');
        }
        return;
      }

      if (q === 0 && decided) {
        var g = svg.select('[data-id="'+nd.id+'"]');
        var rect = g.select('.node-box');
        rect.attr('fill', '#fafafa').attr('stroke', '#ddd').attr('stroke-width', 1);
        g.append('text')
          .attr('class', 'leaf-icon')
          .attr('x', nodeW / 2).attr('y', nodeH / 2 + 4)
          .attr('text-anchor', 'middle')
          .style('font-size', '12px').style('fill', '#ccc')
          .text('\u2717');
        return;
      }

      if (q <= 0) return;
      drawStaticBalls(nd, q);
    });

    svg.selectAll('.tree-edge').each(function() {
      var edge = d3.select(this);
      var pid = edge.attr('data-parent');
      var cid = edge.attr('data-child');
      var parentSplit = animState.splits[pid];
      var childQ = animState.quotas[cid];

      if (parentSplit && childQ !== undefined && childQ > 0) {
        edge.attr('stroke', CW).attr('stroke-width', 2.5).attr('opacity', 0.8);
      } else if (parentSplit && childQ === 0) {
        edge.attr('stroke', '#ddd').attr('stroke-width', 1).attr('opacity', 0.4);
      } else if (parentSplit) {
        edge.attr('stroke', '#ccc').attr('stroke-width', 1.2).attr('opacity', 0.5);
      } else {
        edge.attr('stroke', '#ddd').attr('stroke-width', 1.5).attr('opacity', 1);
      }
    });

    if (animState.done) {
      var selected = [];
      flatNodes.forEach(function(nd) {
        if (nd.leaf && nd.items.length === 1 && animState.quotas[nd.id] === 1)
          selected.push(nd.items[0] + 1);
      });
      d3.select('#sa-result-bar').html(
        '<span style="color:'+CS+'; font-weight:bold;">Sample:</span> $S = \\{' + selected.join(', ') + '\\}$'
      );
      typesetMath();
    } else {
      d3.select('#sa-result-bar').html('');
    }

    updateButtons();
  }

  function isNodeDecided(nd) {
    if (nd.id === tree.root.id) return animState.splits[nd.id] !== undefined;
    if (animState.quotas[nd.id] === undefined) return false;
    for (var qi = 0; qi < flatNodes.length; qi++) {
      var pn = flatNodes[qi];
      if ((pn.left && pn.left.id === nd.id) || (pn.right && pn.right.id === nd.id)) {
        if (animState.splits[pn.id]) return true;
        if (animState.quotas[pn.id] === 0 && isNodeDecided(pn)) return true;
        return false;
      }
    }
    return false;
  }

  function isNodeVisited(nd) {
    if (nd.leaf) return isNodeDecided(nd);
    return animState.splits[nd.id] !== undefined;
  }

  function highlightPMFBar(g, nd, q) {
    var pmfG = g.select('.pmf-hist');
    if (pmfG.empty()) return;
    var pmf = polyToPMF(nd.poly);

    var visited = isNodeVisited(nd);
    var highlightIdx = -1;
    if (visited && q !== undefined && q >= 0 && q < pmf.length) {
      if (nd.leaf) {
        if (q > 0) highlightIdx = q;
      } else {
        highlightIdx = q;
      }
    }

    pmfG.selectAll('rect').each(function(d, i) {
      var isHigh = (i === highlightIdx);
      d3.select(this)
        .attr('fill', isHigh ? CR : CW)
        .attr('opacity', isHigh ? 0.85 : 0.25);
    });
  }

  function buildNodeQueue() {
    var queue = [];
    var levels = tree.levels;
    for (var li = 0; li < levels.length; li++) {
      for (var ni = 0; ni < levels[li].length; ni++) {
        var nd = levels[li][ni];
        if (!nd.pad && !nd.leaf) queue.push(nd);
      }
    }
    return queue;
  }

  var nodeQueue = [];
  var queueIdx = 0;

  function doStep(callback) {
    if (animState.done) { stopPlaying(); if (callback) callback(); return; }

    while (queueIdx < nodeQueue.length) {
      var nd = nodeQueue[queueIdx];
      var q = animState.quotas[nd.id];
      if (q === undefined || q === 0) {
        if (nd.left && !nd.left.pad) animState.quotas[nd.left.id] = 0;
        if (nd.right && !nd.right.pad) animState.quotas[nd.right.id] = 0;
        queueIdx++;
      } else {
        break;
      }
    }

    if (queueIdx >= nodeQueue.length) {
      animState.done = true;
      renderState();
      updateStatusBar();
      stopPlaying();
      if (callback) callback();
      return;
    }

    history.push(snapshot());

    var nd = nodeQueue[queueIdx];
    var k = animState.quotas[nd.id];
    var probs = splitDist(nd.left, nd.right, k);
    var u = uniforms[nd.id];
    var j = sampleWithU(probs, u);
    animState.splits[nd.id] = { probs: probs, j: j, k: k, u: u };
    animState.quotas[nd.left.id] = j;
    animState.quotas[nd.right.id] = k - j;
    animState.level = nd._level;
    queueIdx++;

    showSplitOverlay(nd, false);

    svg.selectAll('.tree-edge').each(function() {
      var edge = d3.select(this);
      if (edge.attr('data-parent') === nd.id) {
        var cid = edge.attr('data-child');
        var childQ = animState.quotas[cid];
        if (childQ !== undefined && childQ > 0) {
          edge.attr('stroke', CW).attr('stroke-width', 2.5).attr('opacity', 0.8);
        } else if (childQ === 0) {
          edge.attr('stroke', '#ddd').attr('stroke-width', 1).attr('opacity', 0.4);
        }
      }
    });

    animating = true;
    updateButtons();

    animateBallSplit([nd], function() {
      animating = false;

      var allDone = true;
      for (var qi = queueIdx; qi < nodeQueue.length; qi++) {
        var remaining = nodeQueue[qi];
        var rq = animState.quotas[remaining.id];
        if (rq !== undefined && rq > 0) { allDone = false; break; }
      }
      if (allDone) {
        for (var qi = queueIdx; qi < nodeQueue.length; qi++) {
          var remaining = nodeQueue[qi];
          if (remaining.left && !remaining.left.pad) animState.quotas[remaining.left.id] = 0;
          if (remaining.right && !remaining.right.pad) animState.quotas[remaining.right.id] = 0;
        }
        queueIdx = nodeQueue.length;
        animState.done = true;
      }

      renderState();
      updateStatusBar();
      showCurrentOverlay();
      if (callback) callback();
    });
  }

  function doBack() {
    if (animating) return;
    stopPlaying();

    if (history.length === 0 && prevRun) {
      restoreSnapshot(prevRun.animState);
      history = prevRun.history;
      uniforms = prevRun.uniforms;
      queueIdx = prevRun.queueIdx;
      prevRun = null;
      renderState();
      updateStatusBar();
      showCurrentOverlay();
      return;
    }

    if (history.length === 0) return;
    restoreSnapshot(history.pop());
    queueIdx = 0;
    for (var qi = 0; qi < nodeQueue.length; qi++) {
      if (animState.splits[nodeQueue[qi].id] === undefined) {
        queueIdx = qi;
        break;
      }
      queueIdx = qi + 1;
    }
    renderState();
    updateStatusBar();
    showCurrentOverlay();
  }

  function showCurrentOverlay() {
    if (animState.done) { clearOverlay(); return; }
    var nextNd = findNextActiveNode();
    if (nextNd) {
      showSplitOverlay(nextNd, true);
    } else {
      clearOverlay();
    }
  }

  function findNextActiveNode() {
    for (var qi = queueIdx; qi < nodeQueue.length; qi++) {
      var nd = nodeQueue[qi];
      var q = animState.quotas[nd.id];
      if (q !== undefined && q > 0) return nd;
    }
    return null;
  }

  function animateBallSplit(nodes, callback) {
    var totalAnim = 0, doneAnim = 0;

    svg.selectAll('.quota-ball').remove();
    svg.selectAll('.quota-num').remove();

    flatNodes.forEach(function(nd) {
      var q = animState.quotas[nd.id];
      if (q === undefined || q <= 0) return;
      var isAnimChild = false;
      nodes.forEach(function(pn) {
        if (pn.left && pn.left.id === nd.id) isAnimChild = true;
        if (pn.right && pn.right.id === nd.id) isAnimChild = true;
        if (pn.id === nd.id) isAnimChild = true;
      });
      if (isAnimChild) return;
      if (nd.leaf && isNodeDecided(nd)) return;
      drawStaticBalls(nd, q);
    });

    nodes.forEach(function(nd) {
      var split = animState.splits[nd.id];
      if (!split) return;
      var k = split.k, j = split.j;

      for (var b = 0; b < k; b++) {
        var goLeft = b < j;
        var target = goLeft ? nd.left : nd.right;
        if (target.pad) continue;
        totalAnim++;

        var gap = ballR * 2 + 2;
        var totalBW = k * gap - 2;
        var startX = nd.x - totalBW / 2 + ballR + b * gap;
        var startY = nd.y;

        var childQ = goLeft ? j : (k - j);
        var childIdx = goLeft ? b : (b - j);
        var endX, endY = target.y;
        if (childQ <= 4) {
          var cTotalBW = childQ * gap - 2;
          endX = target.x - cTotalBW / 2 + ballR + childIdx * gap;
        } else {
          endX = target.x;
        }

        var ball = svg.append('circle')
          .attr('class', 'anim-ball')
          .attr('cx', startX).attr('cy', startY)
          .attr('r', (k <= 4) ? ballR : ballR + 3)
          .attr('fill', CW).attr('opacity', 0.9)
          .attr('stroke', '#fff').attr('stroke-width', 1.5)
          .style('pointer-events', 'none');

        (function(ball, ex, ey) {
          ball.transition()
            .duration(dur)
            .ease(d3.easeCubicInOut)
            .attr('cx', ex).attr('cy', ey)
            .transition()
            .duration(150)
            .attr('opacity', 0)
            .on('end', function() {
              ball.remove();
              doneAnim++;
              if (doneAnim >= totalAnim && callback) callback();
            });
        })(ball, endX, endY);
      }
    });

    if (totalAnim === 0 && callback) callback();
  }

  function drawStaticBalls(nd, q) {
    var cx = nd.x, cy = nd.y;
    if (q <= 4) {
      var gap = ballR * 2 + 2;
      var totalBW = q * gap - 2;
      var sx = cx - totalBW / 2 + ballR;
      for (var i = 0; i < q; i++) {
        svg.append('circle')
          .attr('class', 'quota-ball')
          .attr('cx', sx + i * gap).attr('cy', cy)
          .attr('r', ballR)
          .attr('fill', CW).attr('opacity', 0.85)
          .attr('stroke', '#fff').attr('stroke-width', 1.5)
          .style('pointer-events', 'none');
      }
    } else {
      svg.append('circle')
        .attr('class', 'quota-ball')
        .attr('cx', cx).attr('cy', cy)
        .attr('r', ballR + 3)
        .attr('fill', CW).attr('opacity', 0.85)
        .attr('stroke', '#fff').attr('stroke-width', 1.5)
        .style('pointer-events', 'none');
      svg.append('text')
        .attr('class', 'quota-num')
        .attr('x', cx).attr('y', cy + 4)
        .attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', '#fff').style('font-weight', 'bold')
        .style('pointer-events', 'none')
        .text(q);
    }
  }

  var overlayNodeId = null;
  var overlayIsPreview = null;
  var overlayK = -1;

  function showSplitOverlay(nd, preview) {
    var k, probs, j;
    if (preview) {
      k = animState.quotas[nd.id];
      if (k === undefined || k <= 0) { clearOverlay(); return; }
      probs = splitDist(nd.left, nd.right, k);
      j = -1;
    } else {
      var split = animState.splits[nd.id];
      if (!split || split.k === 0) { clearOverlay(); return; }
      k = split.k; probs = split.probs; j = split.j;
    }

    var barW = 10, barH = 30, gap = 3;
    var maxP = Math.max.apply(null, probs);
    if (maxP === 0) maxP = 1;

    var existing = svg.select('.split-overlay');
    if (!existing.empty() && overlayNodeId === nd.id && overlayIsPreview === preview && overlayK === k) {
      var bars = existing.selectAll('.overlay-bar');
      bars.each(function(d, i) {
        var isSampled = (!preview && i === j);
        var h = Math.max(1, probs[i] / maxP * barH);
        d3.select(this)
          .transition().duration(tweenDur)
          .attr('y', barH - h).attr('height', h)
          .attr('fill', isSampled ? CR : CW)
          .attr('opacity', isSampled ? 0.9 : (preview ? 0.6 : 0.4));
      });
      return;
    }

    svg.selectAll('.split-overlay').remove();
    overlayNodeId = nd.id;
    overlayIsPreview = preview;
    overlayK = k;

    var nBars = probs.length;
    var totalW = nBars * (barW + gap) - gap;
    var cx = nd.x;
    var cy = nd.y + nodeH / 2 + 8;

    var g = svg.append('g')
      .attr('class', 'split-overlay')
      .attr('transform', 'translate(' + (cx - totalW / 2) + ',' + cy + ')')
    g.append('rect')
      .attr('x', -6).attr('y', -4)
      .attr('width', totalW + 12).attr('height', barH + 22)
      .attr('rx', 4)
      .attr('fill', '#fff').attr('stroke', '#ccc')
      .attr('stroke-width', 1)
      .attr('opacity', 0.9);

    for (var jj = 0; jj <= k; jj++) {
      var isSampled = (!preview && jj === j);
      var h = Math.max(1, probs[jj] / maxP * barH);
      var bx = jj * (barW + gap);

      g.append('rect')
        .attr('class', 'overlay-bar')
        .attr('x', bx).attr('y', barH - h)
        .attr('width', barW).attr('height', h)
        .attr('rx', 1.5)
        .attr('fill', isSampled ? CR : CW)
        .attr('opacity', isSampled ? 0.9 : (preview ? 0.6 : 0.4));

      g.append('text')
        .attr('x', bx + barW / 2).attr('y', barH + 11)
        .attr('text-anchor', 'middle')
        .style('font-size', '8px')
        .style('fill', isSampled ? CR : '#aaa')
        .style('font-weight', isSampled ? 'bold' : 'normal')
        .text(jj);
    }
  }

  function clearOverlay() {
    svg.selectAll('.split-overlay').remove();
    overlayNodeId = null;
    overlayIsPreview = null;
    overlayK = -1;
  }

  function showHoverOverlay(nd) {
    hideHoverOverlay();
    var k = animState.quotas[nd.id];
    if (k === undefined) return;
    if (nd.leaf) return;

    var split = animState.splits[nd.id];
    var probs, j;
    if (split) {
      probs = split.probs; j = split.j; k = split.k;
    } else if (k > 0) {
      probs = splitDist(nd.left, nd.right, k);
      j = -1;
    } else {
      return;
    }

    var barW = 10, barH = 30, gap = 3;
    var nBars = probs.length;
    var totalW = nBars * (barW + gap) - gap;
    var cx = nd.x;
    var cy = nd.y - nodeH / 2 - barH - 18;

    var g = svg.append('g')
      .attr('class', 'hover-overlay')
      .attr('transform', 'translate(' + (cx - totalW / 2) + ',' + cy + ')');

    g.append('rect')
      .attr('x', -6).attr('y', -4)
      .attr('width', totalW + 12).attr('height', barH + 22)
      .attr('rx', 4)
      .attr('fill', '#fff').attr('stroke', '#bbb')
      .attr('stroke-width', 1)
      .attr('opacity', 0.95);

    var maxP = Math.max.apply(null, probs);
    if (maxP === 0) maxP = 1;

    for (var jj = 0; jj < nBars; jj++) {
      var isSampled = (jj === j);
      var h = Math.max(1, probs[jj] / maxP * barH);
      var bx = jj * (barW + gap);

      g.append('rect')
        .attr('x', bx).attr('y', barH - h)
        .attr('width', barW).attr('height', h)
        .attr('rx', 1.5)
        .attr('fill', isSampled ? CR : CW)
        .attr('opacity', isSampled ? 0.9 : 0.5);

      g.append('text')
        .attr('x', bx + barW / 2).attr('y', barH + 11)
        .attr('text-anchor', 'middle')
        .style('font-size', '8px')
        .style('fill', isSampled ? CR : '#aaa')
        .style('font-weight', isSampled ? 'bold' : 'normal')
        .text(jj);
    }
  }

  function hideHoverOverlay() {
    svg.selectAll('.hover-overlay').remove();
  }

  function typesetMath() {
    if (window.MathJax && MathJax.typesetPromise) {
      MathJax.typesetPromise().catch(function(){});
    }
  }

  // ── Controls ──
  function stopPlaying() {
    playing = false;
    if (playTimer) { clearTimeout(playTimer); playTimer = null; }
    d3.select('#sa-btn-play').text('Play').classed('active', false);
  }

  function playStep() {
    if (!playing || animState.done) { stopPlaying(); return; }
    doStep(function() {
      if (playing && !animState.done) {
        playTimer = setTimeout(playStep, pauseDur);
      } else {
        stopPlaying();
      }
    });
  }

  d3.select('#sa-btn-back').on('click', function() {
    doBack();
  });

  d3.select('#sa-btn-step').on('click', function() {
    stopPlaying();
    if (animState.done) {
      resetAnimState(false);
      renderState();
    } else {
      doStep();
    }
  });

  d3.select('#sa-btn-play').on('click', function() {
    if (playing) { stopPlaying(); return; }
    if (animState.done) { resetAnimState(false); renderState(); }
    playing = true;
    d3.select('#sa-btn-play').text('Pause').classed('active', true);
    playStep();
  });

  d3.select('#sa-btn-weights').on('click', function() {
    stopPlaying();
    initWeights();
    draw();
  });

  d3.select('#sa-inp-N').on('change', function() {
    var v = Math.max(2, Math.min(16, Math.round(+this.value)));
    if (v === N) return;
    N = v; n = Math.min(n, N);
    d3.select('#sa-inp-n').attr('max', N).property('value', n);
    stopPlaying(); initWeights(); draw();
  });

  d3.select('#sa-inp-n').on('change', function() {
    var v = Math.max(0, Math.min(N, Math.round(+this.value)));
    if (v === n) return;
    var prevQueueIdx = queueIdx;
    var wasDone = animState ? animState.done : false;
    n = v;
    stopPlaying();
    drawScaffolding();
    ensureUniforms();
    replaySplits(prevQueueIdx, wasDone);
    renderState();
    updateStatusBar();
    showCurrentOverlay();
  });

  // ── Init ──
  initWeights();
  draw();

})();
</script>

In pseudocode:

```python
def sample(node, quota):
    if node.is_leaf:
        return [node.item] if quota == 1 else []
    # P_L[j] * P_R[quota-j] for j = 0, ..., quota
    probs = [node.left.poly[j] * node.right.poly[quota - j]
             for j in range(quota + 1)]
    j = categorical(probs)              # how many items from the left subtree
    return sample(node.left, j) + sample(node.right, quota - j)

S = sample(root, n)                     # exactly n items
```

The tree is built once ($\mathcal{O}(N \log^2 N)$) and reused for each sample ($\mathcal{O}(n \log N)$ per sample)—at each of the $\mathcal{O}(\log N)$ levels, only nodes whose quota is nonzero are visited, and there are at most $n$ such nodes.  (When $n \approx N$, nearly all nodes are visited, so the per-sample cost approaches $\mathcal{O}(N \log N)$.)  No $\binom{N}{n}$-sized table is ever constructed.

All computations are verified against brute-force enumeration in the [test suite](test_identities.py).


## Basic Usage

Now that we've seen how the product tree works under the hood, here's the library interface.

```python
from conditional_poisson import ConditionalPoisson
import numpy as np

w = np.array([0.68, 1.02, 0.55, 1.63, 0.67, 2.82])
n = 3

cp = ConditionalPoisson.from_weights(n, w)

cp.log_normalizer       # log Z(w, n)
cp.pi                   # inclusion probabilities (sum to n)
cp.sample(10_000)       # draw 10k exact samples

# Inverse problem: find weights from target inclusion probabilities
cp_fit = ConditionalPoisson.fit(cp.pi, n)
```

## The Poisson Approximation

The inclusion probabilities $\pip_i = P(i \in S)$ always sum to $n$,<a href="test_identities.py#test_pi_sums_to_n" title="test_pi_sums_to_n" class="verified" target="_blank">✓</a> and each $\pip_i \in [0, 1]$.<a href="test_identities.py#test_pi_in_unit_interval" title="test_pi_in_unit_interval" class="verified" target="_blank">✓</a>  Items with larger weights get higher inclusion probabilities, but the relationship is nonlinear—the other weights push back through the size constraint $|S| = n$.

**Poisson approximation.**  If each item were included independently with probability $p_i = \w_i r/(1+\w_i r)$, where $r$ is the **tilting parameter** that makes $\sum_i p_i = n$, we would get

$$\pip_i \;\approx\; \frac{\w_i \, r}{1 + \w_i \, r}.$$

The actual inclusion probabilities are close but not identical—conditioning on $|S|=n$ introduces a correction of $\mathcal{O}(1/N)$ per item (see [Hájek's bound](#Hájek's-approximation) below).

<div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px 20px; margin: 16px 0;">

**Weights vs. inclusion probabilities.** The curve shows the Poisson approximation $p_i = w_i r/(1+w_i r)$; the dots show the exact $\pip_i$ (computed via the product tree).  Drag any dot horizontally to change its weight and see both update.  Use the controls to change $N$ and $n$ or resample weights.

<div id="pi-scatter"></div>
<script>
(function() {
  var N = 10, n = 4;
  var w0 = [0.6799, 1.0196, 0.0198, 0.0023, 0.5503, 1.6299, 0.6736, 0.7553, 2.8168, 6.0578];
  var w = w0.slice();

  function dpPi(w, n) {
    var N = w.length;
    var e = [];
    for (var m = 0; m <= N; m++) e[m] = new Float64Array(n + 1);
    e[0][0] = 1;
    for (var m = 0; m < N; m++) {
      e[m+1][0] = e[m][0];
      for (var k = 1; k <= n; k++)
        e[m+1][k] = e[m][k] + w[m] * e[m][k-1];
    }
    var Z = e[N][n];
    var pi = new Float64Array(N);
    for (var i = 0; i < N; i++) {
      var ei = new Float64Array(n); ei[0] = 1;
      for (var m = 0; m < N; m++) {
        if (m === i) continue;
        for (var k = Math.min(n-1, m < i ? m+1 : m); k >= 1; k--)
          ei[k] += w[m] * ei[k-1];
      }
      pi[i] = w[i] * ei[n-1] / Z;
    }
    return pi;
  }

  function findR(w, n) {
    var r = n / w.reduce(function(a,b){return a+b;}, 0);
    for (var it = 0; it < 100; it++) {
      var f = 0, df = 0;
      for (var i = 0; i < w.length; i++) {
        var wr = w[i] * r, d = 1 + wr;
        f += wr / d; df += w[i] / (d * d);
      }
      var step = (f - n) / df;
      r = Math.max(1e-15, r - step);
      if (Math.abs(step) < 1e-12 * r) break;
    }
    return r;
  }

  function mulberry32(seed) {
    return function() {
      seed |= 0; seed = seed + 0x6D2B79F5 | 0;
      var t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function randExp(rng, N) {
    var w = [];
    for (var i = 0; i < N; i++) w.push(-Math.log(1 - rng()));
    return w;
  }

  var pi;
  var container = d3.select('#pi-scatter');

  var ctrl = container.append('div')
    .style('font-size','0.9em').style('margin-bottom','6px').style('font-family','inherit');
  ctrl.append('span').html('$N$ = ');
  var nInput_N = ctrl.append('input').attr('type','number').attr('min',2).attr('max',50).attr('value',N)
    .style('width','44px').style('font-family','inherit').style('font-size','inherit')
    .style('border','1px solid #ccc').style('border-radius','3px').style('padding','2px 4px')
    .on('change', function(){ var v=+this.value; if(v>=2&&v<=50){N=v; n=Math.min(n,N); nInput_n.property('value',n); newWeights(); build();} });
  ctrl.append('span').html('&ensp;$n$ = ');
  var nInput_n = ctrl.append('input').attr('type','number').attr('min',1).attr('max',N).attr('value',n)
    .style('width','44px').style('font-family','inherit').style('font-size','inherit')
    .style('border','1px solid #ccc').style('border-radius','3px').style('padding','2px 4px')
    .on('change', function(){ var v=+this.value; if(v>=1&&v<=N){n=v; build();} });
  ctrl.append('span').html('&ensp;');
  ctrl.append('button').text('resample weights')
    .style('font-family','inherit').style('font-size','0.85em')
    .style('border','1px solid #ccc').style('border-radius','3px').style('padding','2px 8px')
    .style('cursor','pointer').style('color','#555')
    .on('click', function(){ newWeights(); build(); });

  var seed = 42;
  function newWeights() { seed++; w = randExp(mulberry32(seed), N); }

  var svgEl, gRoot, xAxisG, curvePath, dotG, tipG, tipText;
  var margin = {top: 8, right: 5, bottom: 48, left: 52};
  var W = 510, H = 300;
  var width = W - margin.left - margin.right;
  var height = H - margin.top - margin.bottom;
  var x = d3.scaleLinear().range([0, width]);
  var y = d3.scaleLinear().domain([0, 1]).range([height, 0]);
  var nPts = 200;
  var lineFn = d3.line().x(function(d){return x(d.w);}).y(function(d){return y(d.p);});

  function build() {
    nInput_n.attr('max', N);
    container.selectAll('svg').remove();
    svgEl = container.append('svg').attr('width', W).attr('height', H)
      .style('user-select','none').style('-webkit-user-select','none');
    gRoot = svgEl.append('g').attr('transform','translate('+margin.left+','+margin.top+')');

    xAxisG = gRoot.append('g').attr('transform','translate(0,'+height+')');
    gRoot.append('g').call(d3.axisLeft(y).ticks(5));

    gRoot.append('foreignObject')
      .attr('x', width/2 - 40).attr('y', height + 26).attr('width', 80).attr('height', 24)
      .append('xhtml:div').style('text-align','center').style('font-size','14px')
      .html('$w_i$');

    var yLabelG = gRoot.append('g')
      .attr('transform','translate(-38,'+height/2+') rotate(-90)');
    yLabelG.append('foreignObject')
      .attr('x', -30).attr('y', -10).attr('width', 60).attr('height', 24)
      .append('xhtml:div').style('text-align','center').style('font-size','14px')
      .html('$\\pi_i$');

    curvePath = gRoot.append('path')
      .attr('fill','none').attr('stroke','#5b9bd5').attr('stroke-width',2).attr('opacity',0.8);
    dotG = gRoot.append('g');

    tipG = gRoot.append('g').style('display','none').style('pointer-events','none');
    tipG.append('rect').attr('rx',3).attr('fill','white').attr('stroke','#ccc').attr('opacity',0.92);
    tipText = tipG.append('text').style('font-size','11px').attr('text-anchor','middle').attr('dy','0.35em');

    var lgW = 150, row1Y = 9, row2Y = 27;
    var lg = gRoot.append('g').attr('transform','translate('+(width - lgW + 3)+','+(height-40)+')');
    lg.append('rect').attr('x',-4).attr('y',-3).attr('width',lgW-4).attr('height',40)
      .attr('fill','white').attr('opacity',0.85).attr('rx',3);
    lg.append('line').attr('x1',0).attr('x2',18).attr('y1',row1Y).attr('y2',row1Y)
      .attr('stroke','#5b9bd5').attr('stroke-width',2);
    lg.append('foreignObject').attr('x',24).attr('y',row1Y-10).attr('width',lgW-30).attr('height',22)
      .append('xhtml:div').style('font-size','12px').style('line-height','20px')
      .html('Poisson: $w_i r/(1{+}w_i r)$');
    lg.append('circle').attr('cx',9).attr('cy',row2Y).attr('r',4.5)
      .attr('fill','#E91E63').attr('opacity',0.85);
    lg.append('foreignObject').attr('x',24).attr('y',row2Y-10).attr('width',lgW-30).attr('height',22)
      .append('xhtml:div').style('font-size','12px').style('line-height','20px')
      .html('exact $\\pi_i$');

    redraw();
    var el = container.node();
    if (window.MathJax && MathJax.typesetPromise) {
      MathJax.typesetClear([el]);
      MathJax.typesetPromise([el]);
    }
  }

  function redraw() {
    pi = dpPi(w, n);
    var r = findR(w, n);
    var xMax = Math.max.apply(null, w) * 1.2;
    x.domain([0, xMax]);
    xAxisG.call(d3.axisBottom(x).ticks(6));

    var curveData = [];
    for (var j = 0; j <= nPts; j++) {
      var wj = xMax * j / nPts;
      curveData.push({w: wj, p: wj * r / (1 + wj * r)});
    }
    curvePath.datum(curveData).attr('d', lineFn);

    var dots = dotG.selectAll('.dot').data(d3.range(N));
    dots.exit().remove();
    dots.enter().append('circle').attr('class','dot')
      .attr('r', 5.5).attr('fill','#E91E63').attr('opacity',0.85)
      .style('cursor','ew-resize').style('touch-action','none')
      .on('mouseenter', function(ev,i){ tipG.style('display',null); showTip(i); })
      .on('mouseleave', function(){ tipG.style('display','none'); })
      .call(d3.drag()
        .on('start', function(ev,i){ tipG.style('display',null); })
        .on('drag', function(ev,i){
          w[i] = Math.max(0.001, x.invert(ev.x));
          redraw(); showTip(i);
        })
        .on('end', function(){ tipG.style('display','none'); })
      );
    dotG.selectAll('.dot')
      .attr('cx', function(d){return x(w[d]);})
      .attr('cy', function(d){return y(pi[d]);});
  }

  function showTip(i) {
    var label = 'w='+w[i].toFixed(2)+'  \u03C0='+pi[i].toFixed(3);
    tipText.text(label);
    var bb = tipText.node().getBBox();
    tipG.select('rect').attr('x',bb.x-4).attr('y',bb.y-3)
      .attr('width',bb.width+8).attr('height',bb.height+6);
    var tx = x(w[i]), ty = y(pi[i]) - 16;
    if (ty < 5) ty = y(pi[i]) + 18;
    tipG.attr('transform','translate('+tx+','+ty+')');
  }

  build();
})();
</script>

</div>

**Hájek's approximation.**  [Hájek (1964, Theorem 5.2)](https://doi.org/10.1214/aoms/1177700375) showed that the conditional inclusion probabilities satisfy

$$\pip_i = p_i + \mathcal{O}(1/N)$$

so the Poisson approximation is essentially exact for large $N$.  [Boistard, Lopuhaä & Ruiz-Gazen (2012)](https://arxiv.org/abs/1207.5654) give the explicit correction:

$$\pip_i = p_i\Big(1 - d^{-1}(p_i - \bar{p})(1 - p_i) + \mathcal{O}(d^{-2})\Big)$$

where $d \defeq \sum_i p_i(1-p_i)$ is the variance of the Poisson sample size and $\bar{p} \defeq d^{-1}\sum_i p_i^2(1-p_i)$.  Under the regularity condition $\limsup N/d < \infty$ (which holds whenever the $p_i$ are bounded away from 0 and 1), $d = \Theta(N)$ and the error is $\mathcal{O}(1/N)$ per item.

The correction has a natural interpretation: when $p_i > \bar{p}$ (item $i$ has higher-than-average inclusion probability), conditioning on $|S| = n$ forces the other items to "make room," so $\pip_i > p_i$.  The factor $(1 - p_i)$ modulates this—items already near certain inclusion have less room to adjust.  The denominator $d = \text{Var}(K)$ measures how much slack the system has for redistribution.

We conjecture<a href="test_identities.py#test_poisson_approximation_bound" title="test_poisson_approximation_bound" class="verified" target="_blank">✓</a> a non-asymptotic bound with no hidden constants: $|\pip_i - p_i| \leq p_i(1 - p_i) / d$.  The proof, along with the Boistard et al. derivation, multiple proof attempts, and a summary of what remains open, is in a [companion post](../poisson-approximation-bound/).

**Inverting the approximation.**  Since $p_i = \w_i r/(1+\w_i r)$, we can invert: $\w_i = p_i / (r(1 - p_i))$, giving $\theta_i = \log \w_i = \log(p_i/(1-p_i)) - \log r$.  If we want inclusion probabilities $\bpip^*$ and treat $\pip^*_i \approx p_i$, the initialization $\theta_i^{(0)} = \log(\pip^*_i / (1 - \pip^*_i))$ (i.e., the logit of the target) is off by at most $\mathcal{O}(1/N)$—a good warm start for the [fitting](#Fitting-Weights-to-Target-Probabilities) optimizer.


## Fitting Weights to Target Probabilities

A common use case: you know the inclusion probabilities you *want* and need to find weights that produce them.<a href="test_identities.py#test_fitting_recovers_target" title="test_fitting_recovers_target" class="verified" target="_blank">✓</a>

**Objective.**  The goal is to find the maximum-entropy distribution over size-$n$ subsets whose inclusion probabilities match a target $\bpip^*$.  The primal problem is

$$
\max_{P \in \triangle^{\binom{\mathcal{S}}{n}}} H(P) \quad \text{subject to} \quad \mathbb{E}_P[\mathbf{1}[i \in S]] = \pip^*_i \;\; \forall\, i
$$

where $H(P) \defeq -\sum_S P(S) \log P(S)$ is the Shannon entropy.  We solve this via its dual, which is an unconstrained concave maximization over the log-weights $\btheta$.<footnote>The dual arises by standard exponential-family / Lagrangian duality.  Introduce multipliers $\theta_i$ for each marginal constraint and form the Lagrangian.  The optimal primal distribution has the form $P(S) \propto \exp(\sum_{i \in S} \theta_i)$—exactly the conditional Poisson family—and the dual function to maximize is $L(\btheta) = \bpip^{*\top}\btheta - \log \Zw{\bw}{n}$.  Since the log-partition function $\log \Zw{\bw}{n}$ is convex in $\btheta$, the dual is concave with a unique global maximum.  At the optimum, the inclusion probabilities $\bpip(\btheta)$ match the targets $\bpip^*$ exactly.</footnote>

<details class="derivation">
<summary>Derivation of the dual (click to expand)</summary>

Write the Lagrangian with multipliers $\theta_i$ for each marginal constraint:

$$
\mathcal{L}(P, \btheta) = H(P) + \sum_i \theta_i \big(\mathbb{E}_P[\mathbf{1}[i \in S]] - \pip^*_i\big)
$$

Maximizing over $P$ for fixed $\btheta$ gives the exponential-family form $P^*(S) = \exp\!\big(\sum_{i \in S} \theta_i - \log \Zw{\bw}{n}\big)$, where $\w_i = e^{\theta_i}$—this is precisely the conditional Poisson distribution.  Substituting back yields the dual objective:

$$
L(\btheta) = H(P^*) + \sum_i \theta_i (\pip_i(\btheta) - \pip^*_i) = \bpip^{*\top} \btheta - \log \Zw{\bw}{n}
$$

where the second equality uses $H(P^*) = \log \Zw{\bw}{n} - \bpip(\btheta)^\top \btheta$ (the standard entropy-of-an-exponential-family identity).  Since $\log \Zw{\bw}{n}$ is convex as a log-partition function, $L$ is concave, guaranteeing a unique global maximum.  Strong duality holds because Slater's condition is satisfied (there exists a feasible interior point whenever $0 < \pip^*_i < 1$ and $\sum_i \pip^*_i = n$).

</details>

The log-probability of a subset is $\log P(S) = \sum_{i \in S} \theta_i - \log \Zw{\bw}{n}$, where $\theta_i \defeq \log \w_i$.  The dual objective is:

$$
L(\btheta) \defeq \bpip^{*\top} \btheta - \log \Zw{\bw}{n}
$$

This is concave, guaranteeing a unique global maximum.

**Gradient.**  $\nabla_{\btheta} L(\btheta) = \bpip^* - \bpip(\btheta)$.<a href="test_identities.py#test_fitting_gradient" title="test_fitting_gradient" class="verified" target="_blank">✓</a>  At the optimum, $\bpip(\btheta) = \bpip^*$ exactly, so the gradient is zero.  Each evaluation of $L$ and $\nabla L$ costs $\mathcal{O}(N \log^2 n)$: one pass through the product tree + backpropagation.

**Optimizer.**  L-BFGS converges in a few iterations using only the gradient—no second-order machinery needed.  The [Poisson approximation](#The-Poisson-Approximation) provides a warm start: $\theta_i^{(0)} = \text{logit}(\pip^*_i)$, which has initialization error $\mathcal{O}(1/N)$ per item.

{% notebook conditional-poisson-sampling.ipynb cells[4:5] %}

## Timing

The tree-based approach scales to moderately large $N$ comfortably.  For comparison, the naive $\mathcal{O}(N^2 n)$ dynamic programming (DP) baseline is also shown.  (The PyTorch FFT implementation in the next section is significantly faster—see below.)

{% notebook conditional-poisson-sampling.ipynb cells[5:6] %}

The PyTorch FFT implementation (next section) is faster for three reasons:

1. **FFT-based polynomial multiplication**: $\mathcal{O}(d \log d)$ per multiply instead of $\mathcal{O}(d^2)$, giving $\mathcal{O}(N \log^2 n)$ total with truncation to degree $n$.

2. **Batched execution**: all multiplications at each tree level are batched into a single torch operation ($\mathcal{O}(\log N)$ torch calls, not $\mathcal{O}(N)$).

3. **Autograd**: the gradient (inclusion probabilities) comes from backpropagation—no hand-coded downward pass needed.

The contour scaling (described below) is essential: without it, FFT precision collapses at $N \gtrsim 1000$.  With it, the implementation achieves machine-epsilon accuracy at all tested sizes ($N$ up to 10,000) and handles extreme weight ranges ($r$ from $10^{-10}$ to $10^{8}$).

## PyTorch Implementation with Contour Scaling

The NumPy implementation above uses hand-coded tree traversals with `scipy.signal.convolve`.  A natural question: can we use PyTorch's autograd to compute the gradient (inclusion probabilities) automatically, given only the forward pass?

The answer is yes—and it's both simpler and faster.  The key insight is that **computing $\bpip$ is just backpropagation** applied to the upward pass.  The [Baur-Strassen theorem](https://timvieira.github.io/blog/evaluating-fx-is-as-fast-as-fx/) guarantees that the gradient costs at most a small constant factor more than the forward pass ([Baur & Strassen, 1983](https://doi.org/10.1016/0304-3975(83)90110-X) prove $3\times$ for nonscalar operations on polynomials; [Griewank & Walther, 2008](https://doi.org/10.1137/1.9780898717761) give $5\times$ for general reverse-mode AD).  [Griewank & Walther (2008)](https://doi.org/10.1137/1.9780898717761) show that the numerical stability of the derivatives is inherited from the forward pass—so if we make the forward pass stable, everything else follows.

The computation is just the product tree: build $\prod_i (1 + \w_i \z)$ bottom-up using polynomial multiplication, then extract $\llbracket \cdot \rrbracket(\z^n)$ and take the log.  In PyTorch, we batch all multiplications at each tree level into a single operation—$\mathcal{O}(\log N)$ torch calls instead of $\mathcal{O}(N)$.

### The FFT Precision Problem and Weight Rescaling

Using FFT for the polynomial multiplications gives $\mathcal{O}(N \log^2 n)$ complexity (with truncation to degree $n$).  But naively, FFT introduces rounding errors $\approx \varepsilon \cdot \max_k|c_k|$ per coefficient, where $c_k$ denotes the $k$<sup>th</sup> coefficient of the product polynomial $\prod_i(1 + \w_i \z) = \sum_k c_k \z^k$.  The largest coefficient (near degree $N/2$) can be $\approx 10^{300}$ times larger than $c_n$, the coefficient we need—so FFT noise drowns the signal.

**The key identity.**  For any $r > 0$:<a href="test_identities.py#test_contour_scaling" title="test_contour_scaling" class="verified" target="_blank">✓</a>

$$\Zw{\bw}{n} = r^{-n} \cdot \llbracket \textstyle\prod_i(1 + \w_i r\, \z) \rrbracket(\z^n)$$

This is immediate: each $\z^n$ term in the product picks up a factor of $r$ per item, giving $r^n \cdot \Zw{\bw}{n}$.  The identity says we can rescale all weights by any $r > 0$ and recover the same answer.  In exact arithmetic, $r$ doesn't matter.  In floating-point, it matters enormously: different values of $r$ shift which coefficient is largest, changing the dynamic range and thus the FFT rounding error relative to $c_n$.

**Choosing $r$.**  The FFT error in $c_n$ is $\mathcal{O}(\varepsilon \cdot \max_k |c_k|)$, so the relative error is $\max_k |c_k| / |c_n|$ times machine epsilon.  If $c_n$ is the largest coefficient, the relative error is just $\mathcal{O}(\varepsilon)$.  After rescaling by $r$, the $k$<sup>th</sup> coefficient becomes $c_k(r) = \Zw{\bw}{k} \cdot r^k$.  We want $r$ such that $c_n(r) \approx \max_k c_k(r)$.

This is where the [Poisson approximation](#The-Poisson-Approximation) reappears.  Define $p_i \defeq \w_i r/(1 + \w_i r)$—the same Poisson inclusion probabilities from earlier.  Then $c_k(r) = \prod_i(1+\w_i r) \cdot \Pr[K = k]$ where $K \defeq \sum_i \text{Bernoulli}(p_i)$ and $\prod_i(1+\w_i r)$ is independent of $k$.  So $\arg\max_k c_k(r) = \text{mode}(K)$.  For a sum of independent Bernoullis, $|\text{mode}(K) - \mathbb{E}[K]| \le 1$ ([Darroch, 1964](https://doi.org/10.1214/aoms/1177703287)).  Setting $\mathbb{E}[K] = n$ places the mode at degree $n \pm 1$, which means the optimal contour radius is exactly the tilting parameter from the Poisson approximation:<a href="test_identities.py#test_contour_r_solves_expected_size" title="test_contour_r_solves_expected_size" class="verified" target="_blank">✓</a>

$$\sum_i \frac{\w_i \cdot r}{1 + \w_i \cdot r} = \sum_i p_i = n$$

In other words, the $r$ that makes the Poisson approximation $\pip_i \approx p_i$ satisfy $\sum p_i = n$ is the same $r$ that minimizes FFT rounding error.  This is not a coincidence: both are asking for the rescaling that centers the Poisson Binomial distribution at degree $n$.

This is monotone in $\log r$ (the LHS increases from 0 to $N$), so Newton's method converges in a few iterations.  A simpler heuristic, $r = n / \W$ (where $\W \defeq \sum_i \w_i$), linearizes $\w_i r / (1 + \w_i r) \approx \w_i r$ and works for mild weights but breaks down with heavy tails.

<div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px 20px; margin: 16px 0;">

**Contour diagram.** The generating function $f(\z) = \prod_i(1 + \w_i \z)$ has zeros at $\z = -1/\w_i$ on the negative real axis.  Extracting the $n$<sup>th</sup> coefficient via FFT is equivalent to sampling $f$ at equally-spaced points on the circle $|\z| = r$.  Drag the weight sliders or the circle to explore.

<div id="contour-diagram"></div>
<script>
(function() {
  var W = 500, H = 280, barW = 500, barH = 100;
  var WMAX = 6;
  var CI = '#d4a24e', CW = '#5b9bd5', CR = '#c0504d';
  var w = [1.5, 3.2, 0.8, 4.5, 2.0, 0.3, 5.1, 1.8];
  var N = w.length, n = 3;
  var r;

  function solveR(target) {
    var lr = 0;
    for (var i = 0; i < 50; i++) {
      var rv = Math.exp(lr);
      var s = 0, ds = 0;
      for (var j = 0; j < N; j++) {
        var p = w[j]*rv/(1+w[j]*rv);
        s += p; ds += p*(1-p);
      }
      lr += (target - s) / ds;
    }
    return Math.exp(lr);
  }

  function coeffs(rv) {
    var poly = [1];
    for (var i = 0; i < N; i++) {
      var next = new Array(poly.length + 1).fill(0);
      for (var j = 0; j < poly.length; j++) {
        next[j] += poly[j];
        next[j+1] += poly[j] * w[i] * rv;
      }
      poly = next;
    }
    return poly;
  }

  var root = d3.select('#contour-diagram');

  // Weight sliders
  var sliderH = 60, sliderW = 22;
  var sliderSvg = root.append('svg')
    .attr('width', N * (sliderW + 6) + 40).attr('height', sliderH + 20)
    .style('display','block').style('margin','0 auto 4px auto');
  var sliderFills = [], sliderLabels = [];
  for (var i = 0; i < N; i++) {
    (function(idx) {
      var sx = 10 + idx * (sliderW + 6);
      var sg = sliderSvg.append('g');
      // Track
      sg.append('rect').attr('x', sx + 1).attr('y', 14)
        .attr('width', sliderW - 2).attr('height', sliderH)
        .attr('fill', '#f8f8f8').attr('stroke', '#eee').attr('rx', 1);
      // Fill
      var frac = Math.min(w[idx] / WMAX, 1);
      var sf = sg.append('rect')
        .attr('x', sx + 2).attr('width', sliderW - 4).attr('rx', 1)
        .attr('y', 14 + sliderH - frac * sliderH).attr('height', frac * sliderH)
        .attr('fill', CI).attr('opacity', 0.8).style('pointer-events', 'none');
      sliderFills.push(sf);
      // Label
      var sl = sg.append('text')
        .attr('x', sx + sliderW/2).attr('y', 14 + sliderH - frac * sliderH - 2)
        .attr('text-anchor', 'middle')
        .style('font-size', '9px').style('fill', CI).style('pointer-events', 'none')
        .text(w[idx].toFixed(1));
      sliderLabels.push(sl);
      // Drag target
      sg.append('rect')
        .attr('x', sx - 2).attr('y', 12)
        .attr('width', sliderW + 4).attr('height', sliderH + 4)
        .attr('fill', 'transparent').attr('cursor', 'ns-resize')
        .style('touch-action', 'none')
        .call(d3.drag().on('drag', function(event) {
          var fr = (14 + sliderH - event.y) / sliderH;
          w[idx] = Math.max(0.01, Math.min(WMAX, fr * WMAX));
          build();
        }));
    })(i);
  }

  function updateSliders() {
    for (var i = 0; i < N; i++) {
      var frac = Math.min(w[i] / WMAX, 1);
      sliderFills[i].attr('y', 14 + sliderH - frac * sliderH).attr('height', frac * sliderH);
      sliderLabels[i].attr('y', 14 + sliderH - frac * sliderH - 2).text(w[i].toFixed(1));
    }
  }

  // Complex plane + bar chart containers
  var svg, cx, cy, pxPerUnit, contour, zeroDots, rLabel, rStarLabel, dots, rOpt;
  var barSvg, barG, bars, barLabels, bw, dynLabel;
  var nFFT = 16;

  function toX(v) { return cx + v * pxPerUnit; }
  function toY(v) { return cy - v * pxPerUnit; }

  function build() {
    updateSliders();
    var zeros = w.map(function(wi){ return -1/wi; });
    rOpt = solveR(n);
    r = rOpt;

    // Remove old SVGs (keep slider SVG)
    root.selectAll('svg:not(:first-child)').remove();

    // Complex plane
    var minZero = Math.min.apply(null, zeros);
    var viewLeft = minZero * 1.3;
    var viewRight = -viewLeft * 0.5;
    cx = W * 0.5; cy = H * 0.5;
    pxPerUnit = W / (viewRight - viewLeft);

    svg = root.append('svg').attr('width', W).attr('height', H)
      .style('display','block').style('margin','0 auto 8px auto');

    // Axes
    svg.append('line').attr('x1',0).attr('x2',W).attr('y1',cy).attr('y2',cy)
      .attr('stroke','#ccc').attr('stroke-width',1);
    svg.append('line').attr('x1',toX(0)).attr('x2',toX(0)).attr('y1',0).attr('y2',H)
      .attr('stroke','#ccc').attr('stroke-width',1);
    svg.append('text').attr('x',W-8).attr('y',cy-6).attr('text-anchor','end')
      .style('font-size','12px').style('fill','#999').style('font-family',"'EB Garamond', serif").text('Re');
    svg.append('text').attr('x',toX(0)+8).attr('y',14)
      .style('font-size','12px').style('fill','#999').style('font-family',"'EB Garamond', serif").text('Im');

    // Contour circle
    contour = svg.append('circle')
      .attr('cx', toX(0)).attr('cy', cy)
      .attr('fill','rgba(90,155,213,0.08)').attr('stroke','#4a90d9').attr('stroke-width',2)
      .attr('stroke-dasharray','6,3');

    // Zeros
    zeroDots = [];
    zeros.forEach(function(z) {
      zeroDots.push(svg.append('circle').attr('cx', toX(z)).attr('cy', cy).attr('r', 5)
        .attr('fill','none').attr('stroke', CR).attr('stroke-width',2));
    });
    svg.append('text').attr('x', toX(zeros[0])).attr('y', cy + 20)
      .attr('text-anchor','middle')
      .style('font-size','11px').style('fill', CR).style('font-family',"'EB Garamond', serif")
      .text('zeros: −1/wᵢ');

    // Pole at origin
    svg.append('text').attr('x', toX(0)-3).attr('y', cy-3).attr('text-anchor','end')
      .style('font-size','16px').style('fill','#333').text('×');

    rLabel = svg.append('text').attr('text-anchor','start')
      .style('font-size','13px').style('fill','#4a90d9').style('font-family',"'EB Garamond', serif");
    rStarLabel = svg.append('text').attr('text-anchor','start')
      .style('font-size','11px').style('fill','#999').style('font-family',"'EB Garamond', serif");

    dots = svg.selectAll('.fft-dot').data(d3.range(nFFT)).enter()
      .append('circle').attr('class','fft-dot').attr('r',2.5)
      .attr('fill','#4a90d9').attr('opacity',0.5);

    // Drag on circle
    contour.call(d3.drag().on('drag', function(event) {
      var dx = event.x - toX(0), dy = event.y - cy;
      r = Math.max(0.05, Math.sqrt(dx*dx + dy*dy) / pxPerUnit);
      update();
    })).style('cursor','ew-resize');
    svg.append('circle').attr('cx',toX(0)).attr('cy',cy)
      .attr('r', W/2).attr('fill','transparent').attr('pointer-events','all')
      .style('cursor','ew-resize')
      .call(d3.drag().on('drag', function(event) {
        var dx = event.x - toX(0), dy = event.y - cy;
        r = Math.max(0.05, Math.sqrt(dx*dx + dy*dy) / pxPerUnit);
        update();
      }));

    // Bar chart
    barSvg = root.append('svg').attr('width', barW).attr('height', barH + 30)
      .style('display','block').style('margin','0 auto');
    barG = barSvg.append('g').attr('transform','translate(40,5)');
    bw = (barW - 80) / (N + 1);
    bars = barG.selectAll('.coeff-bar').data(d3.range(N+1)).enter()
      .append('rect').attr('class','coeff-bar')
      .attr('x', function(d){ return d * bw; })
      .attr('width', bw - 2).attr('fill', function(d){ return d === n ? CR : CW; })
      .attr('opacity', function(d){ return d === n ? 0.9 : 0.5; });
    barLabels = barG.selectAll('.coeff-label').data(d3.range(N+1)).enter()
      .append('text').attr('class','coeff-label')
      .attr('x', function(d){ return d * bw + (bw-2)/2; })
      .attr('text-anchor','middle')
      .style('font-size','10px').style('fill','#666').style('font-family',"'EB Garamond', serif");
    barG.selectAll('.k-label').data(d3.range(N+1)).enter()
      .append('text').attr('class','k-label')
      .attr('x', function(d){ return d * bw + (bw-2)/2; })
      .attr('text-anchor','middle')
      .style('font-size','11px').style('fill','#333').style('font-family',"'EB Garamond', serif")
      .text(function(d){ return d; });
    barSvg.append('text').attr('x', barW/2).attr('y', barH + 26)
      .attr('text-anchor','middle')
      .style('font-size','12px').style('fill','#666').style('font-family',"'EB Garamond', serif")
      .text('degree k');
    dynLabel = barSvg.append('text').attr('x', barW - 10).attr('y', 16)
      .attr('text-anchor','end')
      .style('font-size','11px').style('fill','#999').style('font-family',"'EB Garamond', serif");

    update();
  }

  function update() {
    var rPx = r * pxPerUnit;
    contour.attr('r', rPx);
    dots.attr('cx', function(d){ return toX(r * Math.cos(2*Math.PI*d/nFFT)); })
        .attr('cy', function(d){ return toY(r * Math.sin(2*Math.PI*d/nFFT)); });
    rLabel.attr('x', toX(r) + 6).attr('y', cy - 4).text('r = ' + r.toFixed(2));
    rStarLabel.attr('x', toX(rOpt) + 4).attr('y', cy + 24).text('r* = ' + rOpt.toFixed(2));

    var c = coeffs(r);
    var maxC = Math.max.apply(null, c);
    var barMax = barH - 10;
    bars.attr('y', function(d){ return barMax - (c[d]/maxC) * barMax + 5; })
        .attr('height', function(d){ return (c[d]/maxC) * barMax; });
    barLabels.attr('y', function(d){ return barMax - (c[d]/maxC) * barMax; })
      .text(function(d){ var v = c[d]/maxC; return v > 0.05 ? v.toFixed(2) : ''; });
    barG.selectAll('.k-label').attr('y', barMax + 16);
    var dynRange = maxC / Math.max(c[n], 1e-300);
    dynLabel.text('max/cₙ = ' + (dynRange < 100 ? dynRange.toFixed(1) : dynRange.toExponential(1)));
  }

  build();
})();
</script>

</div>

**Guarantee.** With this choice of $r$, the coefficients $c_k(r)$ are proportional to the PMF of $K$, whose mode is at $n \pm 1$.  The PMF of a sum of Bernoullis is log-concave, so it decays exponentially away from the mode.  In practice, this makes the dynamic range $\max_k c_k / c_n$ a modest constant (typically $\approx 1$), ensuring the relative FFT error in $c_n$ is $\mathcal{O}(\varepsilon)$.


**Autograd compatibility.** The rescaling $\w_i \mapsto \w_i \cdot r$ is on the autograd graph (it's just a scalar multiply); the root-finding for $r$ is not (it's a numerical conditioning choice, not part of the mathematical function).  Gradients flow through $\w_i \cdot r$ as if $r$ were a constant—which is correct, since $\log \Z$ does not depend on $r$ (every $r$ gives the same answer in exact arithmetic).


<div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px 20px; margin: 16px 0;">

**Numerical validation.** The table below shows the dynamic range ($\max_k |c_k| / |c_n|$) and $\log \Z$ error for three choices of $r$ on two weight regimes ($N=200$, $n=10$).  With $r=1$ the dynamic range is $\approx 10^{16}$, destroying all precision.  The optimal $r^*$ brings it to $\approx 1$.

<div id="r-comparison"></div>
<script>
(function() {
  var data = [
    {label:'r = 1',   dr:1e16, err:8e+01},
    {label:'r = n/W', dr:1e0,  err:7e-15},
    {label:'r = r*',  dr:1e0,  err:1e-14},
    {label:'r = 1',   dr:1e16, err:2e+02},
    {label:'r = n/W', dr:8e1,  err:2e-13},
    {label:'r = r*',  dr:1e0,  err:4e-14},
  ];

  var groups = [
    {title: 'mild weights', rows: data.slice(0,3)},
    {title: 'heavy tails',  rows: data.slice(3,6)},
  ];

  var container = d3.select('#r-comparison');
  var margin = {top: 4, right: 100, bottom: 30, left: 70};
  var W = 500, rowH = 24, groupGap = 18, groupLabelH = 18;
  var totalH = groups.length * (3 * rowH + groupLabelH + groupGap) - groupGap;
  var H = margin.top + margin.bottom + totalH;
  var width = W - margin.left - margin.right;

  var svg = container.append('svg').attr('width', W).attr('height', H)
    .style('user-select','none').style('-webkit-user-select','none');
  var g = svg.append('g').attr('transform','translate('+margin.left+','+margin.top+')');

  var x = d3.scaleLog().domain([1, 1e17]).range([0, width]).clamp(true);

  // x-axis at bottom
  g.append('g').attr('transform','translate(0,'+totalH+')')
    .call(d3.axisBottom(x).ticks(6, '.0e').tickSize(3))
    .selectAll('text').style('font-size','10px');
  g.append('text').attr('x', width/2).attr('y', totalH + 26)
    .attr('text-anchor','middle').style('font-size','11px').style('fill','#666')
    .text('dynamic range (log scale)');

  var colors = {'r = 1':'#c0504d', 'r = n/W':'#999', 'r = r*':'#5b9bd5'};
  var yPos = 0;

  groups.forEach(function(grp, gi) {
    // Group title
    g.append('text').attr('x', -6).attr('y', yPos + 12)
      .attr('text-anchor','end').style('font-size','11px')
      .style('fill','#888').style('font-style','italic')
      .text(grp.title);
    yPos += groupLabelH;

    grp.rows.forEach(function(d) {
      var barW = Math.max(2, x(Math.max(1.01, d.dr)));
      // Bar
      g.append('rect').attr('x', 0).attr('y', yPos + 3)
        .attr('width', barW).attr('height', rowH - 6)
        .attr('fill', colors[d.label]).attr('opacity', 0.8).attr('rx', 2);
      // Row label
      g.append('text').attr('x', -6).attr('y', yPos + rowH/2 + 1)
        .attr('text-anchor','end').attr('dominant-baseline','middle')
        .style('font-size','12px').style('fill','#333')
        .text(d.label);
      // Dynamic range value on bar
      var drText = d.dr >= 10 ? '10' + '\u2070\u00B9\u00B2\u00B3\u2074\u2075\u2076\u2077\u2078\u2079'.split('').slice(0,0) : '';
      // simpler: just use text
      var drStr = d.dr >= 1e3 ? '10^' + Math.round(Math.log10(d.dr)) : d.dr.toFixed(0);
      var errStr = d.err.toExponential(0);
      g.append('text').attr('x', barW + 5).attr('y', yPos + rowH/2 + 1)
        .attr('dominant-baseline','middle')
        .style('font-size','11px').style('fill','#666')
        .text('DR=' + drStr + ',  |err|=' + errStr);
      yPos += rowH;
    });
    yPos += groupGap;
  });
})();
</script>

</div>

Code: [`torch_fft_prototype.py`](https://github.com/timvieira/conditional-poisson-sampling/blob/main/torch_fft_prototype.py)

## Application: Horvitz-Thompson Estimation

So far we've built machinery for sampling fixed-size subsets and computing inclusion probabilities.  A natural question: what can you *do* with this?  One important application is unbiased estimation.

**Setup.** Suppose you have a distribution $p$ over a universe $\mathcal{S}$ of $N$ items, and a function $f$ that is expensive to evaluate.  You want to estimate $\mu = \sum_i p(i)\, f(i)$ using only $n$ evaluations of $f$.  With i.i.d. Monte Carlo, you'd draw $n$ samples—but some items may repeat, wasting evaluations.

**The estimator.** The **Horvitz-Thompson estimator** ([Horvitz & Thompson, 1952](https://doi.org/10.1080/01621459.1952.10483446)) uses sampling *without* replacement to guarantee $n$ *distinct* evaluations.  Draw a fixed-size subset $S \sim P_n$ using conditional Poisson sampling, then form:

$$
\hat{\mu}_{\text{HT}}(S) = \sum_{i \in S} \frac{p(i)}{\pip_i}\, f(i), \quad S \sim P_n
$$

This gives an unbiased estimate: $\mathbb{E}[\hat{\mu}_{\text{HT}}] = \mu$,<a href="test_identities.py#test_horvitz_thompson_unbiased" title="test_horvitz_thompson_unbiased" class="verified" target="_blank">✓</a> provided $\pip_i > 0$ whenever $p(i) > 0$.  The inverse-probability weighting $p(i)/\pip_i$ corrects for the sampling bias—items with higher inclusion probability are down-weighted, and vice versa.

**Example.** With $N = 100$ items and $n = 5$, set weights proportional to $p(i)$ so that high-probability items are more likely to be selected.  Each sample gives 5 distinct evaluations of $f$; the HT formula reweights them to produce an unbiased estimate of the full sum.

For more on SWOR-based estimation (including the near-optimal priority sampling scheme), see my earlier post on [estimating means in a finite universe](https://timvieira.github.io/blog/post/2017/07/03/estimating-means-in-a-finite-universe/).

## Identities for $\Z$ and Its Relatives

The normalizing constant $\Zw{\bw}{n}$ is the $n$<sup>th</sup> elementary symmetric polynomial $e_n(\bw)$.  Here are some useful identities.


### Differential Identities

The exponential family structure gives the inclusion probability as the gradient of the log-normalizer:

$$\pip_i \defeq \frac{\partial \log \Z}{\partial \theta_i} = \frac{\w_i \cdot \Zw{\bw^{(-i)}}{n-1}}{\Zw{\bw}{n}}$$
<a href="test_identities.py#test_pi_is_gradient_of_log_Z" title="test_pi_is_gradient_of_log_Z, test_pi_leave_one_out, test_pi_matches_brute_force" class="verified" target="_blank">✓</a>

The leave-one-out formula generalizes to **higher-order inclusion probabilities**: $\pip(X) = P(X \subseteq S) = \prod_{i \in X} \w_i \cdot \Zw{\bw^{(-X)}}{n-|X|} / \Zw{\bw}{n}$.<a href="test_identities.py#test_higher_order_inclusion" title="test_higher_order_inclusion" class="verified" target="_blank">✓</a>

### Recurrences and Algorithms

**Weighted Pascal recurrence.** The $\mathcal{O}(Nn)$ dynamic programming algorithm for $\Z$:

$$\Zw{\w_1, \ldots, \w_m}{k} = \Zw{\w_1, \ldots, \w_{m-1}}{k} + \w_m \cdot \Zw{\w_1, \ldots, \w_{m-1}}{k-1}$$

with base cases $\Zw{\cdot}{0} = 1$ and $\Zw{\cdot}{k} = 0$ for $k < 0$ or $k > m$.  This is a weighted generalization of Pascal's identity $\binom{m}{k} = \binom{m-1}{k} + \binom{m-1}{k-1}$.  Include item $m$ (second term) or exclude it (first term).<a href="test_identities.py#test_weighted_pascal_recurrence" title="test_weighted_pascal_recurrence" class="verified" target="_blank">✓</a>


Running this DP forward gives $\Z$; running backpropagation on it gives $\bpip$—all in $\mathcal{O}(Nn)$.  This is simpler than the product tree for moderate $N$, and illustrates the same principle: the gradient comes for free via automatic differentiation.

**Proposition (Weighted Vandermonde identity).** For disjoint groups $A, B$ with weight vectors $\ba$ and $\bb$:

$$\Zw{(\ba;\, \bb)}{k} = \sum_{j=0}^{k} \Zw{\ba}{j} \cdot \Zw{\bb}{k-j}$$

This is a weighted generalization of [Vandermonde's identity](https://en.wikipedia.org/wiki/Vandermonde%27s_identity) $\binom{a+b}{k} = \sum_{j=0}^{k} \binom{a}{j}\binom{b}{k-j}$.<a href="test_identities.py#test_weighted_vandermonde" title="test_weighted_vandermonde" class="verified" target="_blank">✓</a>  The proof is immediate from the generating function: $\Zw{\bw}{k}$ is the $k$<sup>th</sup> coefficient of $\prod_i(1 + \w_i \z)$, and factoring this product over $A \cup B$ turns coefficient extraction into a convolution.

<details class="derivation">
<summary>Proof</summary>

Factor the generating function over $A \cup B$:

$$\prod_{i \in A \cup B}(1 + \w_i \z) = \prod_{i \in A}(1 + \w_i \z) \;\cdot\; \prod_{i \in B}(1 + \w_i \z)$$

The left side is $\sum_k \Zw{(\ba;\,\bb)}{k}\, \z^k$.  The right side is a product of two polynomials, so its $k$<sup>th</sup> coefficient is the convolution $\sum_{j=0}^{k} \Zw{\ba}{j} \cdot \Zw{\bb}{k-j}$.  Equating coefficients of $\z^k$ gives the result. <span style="float:right">$\blacksquare$</span>

</details>

This is why polynomial multiplication computes $\Z$: the product tree exploits this identity at every node.  It is also what makes the sampling algorithm correct—splitting a quota of $k$ items between two subtrees according to $\Zw{\ba}{j} \cdot \Zw{\bb}{k-j}$ produces the exact conditional distribution.


**Newton's identities.** The elementary symmetric polynomials can be computed from **power sums** $g_k \defeq \sum_i \w_i^k$ (see [Stanley (1999)](https://doi.org/10.1017/CBO9780511609589), Chapter 7):

$$\Zw{\bw}{k} = \sum_{i=1}^{k} \frac{(-1)^{i-1}}{k}\, \Zw{\bw}{k-i} \cdot g_i$$

<a href="test_identities.py#test_newtons_identities" title="test_newtons_identities" class="verified" target="_blank">✓</a> This is an $\mathcal{O}(Nn)$ algorithm that only needs the power sums, not the individual weights—useful when the universe is implicitly defined (e.g., paths in a weighted finite-state automaton, where $g_k$ can be computed via matrix methods).

### Connection to K-DPPs

<details class="derivation">
<summary>A diagonal K-DPP is exactly a conditional Poisson distribution (click to expand)</summary>

A $K$-DPP (fixed-size determinantal point process) on $\{1, \ldots, N\}$ with a **diagonal** kernel matrix $L = \text{diag}(\w_1, \ldots, \w_N)$ is exactly the conditional Poisson distribution:

$$\mathcal{P}_L^K(S) = \frac{\det(L_S)}{\sum_{|S'|=K} \det(L_{S'})} = \frac{\prod_{i \in S} \w_i}{\Zw{\bw}{K}}$$

since the determinant of a diagonal submatrix is the product of its diagonal entries.<a href="test_identities.py#test_kdpp_diagonal" title="test_kdpp_diagonal" class="verified" target="_blank">✓</a>  The normalizer $\sum_{|S|=K} \det(L_S) = e_K(\lambda_1, \ldots, \lambda_N)$ is an elementary symmetric polynomial of the eigenvalues—which for a diagonal matrix are just the weights.  Newton's identities connect the two: compute $e_K$ from the power sums $g_k = \text{tr}(L^k)$.

For non-diagonal $L$, the K-DPP introduces correlations between items (repulsion), while the conditional Poisson distribution has only the size constraint.  See [Kulesza & Taskar (2012)](https://arxiv.org/abs/1207.6083) for details.

</details>

## Summary

**The takeaway.** A single data structure—the polynomial product tree—unifies normalizing constant computation, marginal inference, sampling, and parameter fitting for the conditional Poisson distribution.  Each capability is a mechanical program transformation of the previous one:

| | What | How | Cost |
|---|---|---|---|
| $\Z$ | normalizing constant | forward pass (polynomial multiply) | $\mathcal{O}(N \log^2 n)$ |
| $\bpip$ | inclusion probabilities | backward pass (reverse-mode AD) | $\mathcal{O}(N \log^2 n)$ |
| $S$ | exact samples | top-down quota splitting (Vandermonde) | $\mathcal{O}(n \log N)$ per sample |
| $\btheta^*$ | fitted parameters | L-BFGS (gradient only) | $\mathcal{O}(N \log^2 n)$ per iteration |

There are no problem-specific derivations here—each row follows from a general theorem in automatic differentiation or computer algebra.

The NumPy implementation ([`conditional_poisson.py`](https://github.com/timvieira/conditional-poisson-sampling/blob/main/conditional_poisson.py)) uses hand-coded tree traversals with $\mathcal{O}(N \log^2 n)$ complexity.  The PyTorch implementation ([`torch_fft_prototype.py`](https://github.com/timvieira/conditional-poisson-sampling/blob/main/torch_fft_prototype.py)) uses FFT-based polynomial multiplication with **contour radius scaling**—rescaling weights to shift the product polynomial's peak to degree $n$, making FFT numerically stable—achieving $\mathcal{O}(N \log^2 n)$ with full autograd support.

**References:**

- [Hájek (1964)](https://doi.org/10.1214/aoms/1177700375). "Asymptotic Theory of Rejective Sampling with Varying Probabilities from a Finite Population." *The Annals of Mathematical Statistics*, 35(4), 1491–1523.

- [Jaynes (1957)](https://doi.org/10.1103/PhysRev.106.620). "Information Theory and Statistical Mechanics." *Physical Review*, 106(4), 620–630.

- [Chen, Dempster & Liu (1994)](https://academic.oup.com/biomet/article-abstract/81/3/457/256956). "Weighted Finite Population Sampling to Maximize Entropy." *Biometrika*, 81(3), 457–469.

- [Horvitz & Thompson (1952)](https://doi.org/10.1080/01621459.1952.10483446). "A Generalization of Sampling Without Replacement from a Finite Universe." *Journal of the American Statistical Association*, 47(260), 663–685.

- [Darroch (1964)](https://doi.org/10.1214/aoms/1177703287). "On the Distribution of the Number of Successes in Independent Trials." *The Annals of Mathematical Statistics*, 35(3), 1317–1321.
- [Baur & Strassen (1983)](https://doi.org/10.1016/0304-3975(83)90110-X). "The Complexity of Partial Derivatives." *Theoretical Computer Science*, 22(3), 317–330.


- [Griewank & Walther (2008)](https://doi.org/10.1137/1.9780898717761). *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*, 2nd edition. SIAM.

- [Stanley (1999)](https://doi.org/10.1017/CBO9780511609589). *Enumerative Combinatorics*, Volume 2. Cambridge University Press.

- [von zur Gathen & Gerhard (2013)](https://doi.org/10.1017/CBO9781139856065). *Modern Computer Algebra*, 3rd edition. Cambridge University Press.

- [Tillé (2006)](https://link.springer.com/book/10.1007/978-0-387-34240-0). *Sampling Algorithms*. Springer.

- [Meister, Amini, Vieira & Cotterell (2021)](https://aclanthology.org/2021.emnlp-main.52/). "Conditional Poisson Stochastic Beams." *Proceedings of EMNLP 2021*.