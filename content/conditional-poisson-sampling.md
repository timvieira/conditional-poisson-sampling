title: Conditional Poisson Sampling
date: 2026-03-25
comments: true
tags: notebook, sampling, algorithms, sampling-without-replacement

{% notebook conditional-poisson-sampling.ipynb cells[1:5] %}

<style>
#cps table { border-collapse: collapse; width: 100%; margin: 0; table-layout: fixed; }
#cps th, #cps td { padding: 1px 2px; font-family: inherit; font-size: 0.85em; line-height: 1.3; }
#cps th { border-bottom: 1px solid #ccc; font-weight: normal; color: #666; }
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
  var N=5, n=3, w=[0.124,0.265,0.066,0.372,0.174], pi=[];
  function normW(){var s=w.reduce(function(a,b){return a+b;},0);w=w.map(function(v){return v/s;});}
  normW();
  var CW='#5b9bd5', CP='#c0504d';
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
    // Normalize w to sum to 1 (scale-invariant, keeps values in [0,1])
    var ww = th.map(function(t){return Math.exp(t);});
    var s = ww.reduce(function(a,b){return a+b;},0);
    return ww.map(function(v){return v/s;});
  }

  function getMaxW(){ return 1; }

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
      .on('change input',function(){var v=+this.value;if(isNaN(v))return;if(v<2||v>8){d3.select('#cps-status').text('the distribution requires 2 \u2264 N \u2264 8').style('color','#c00');return;}v=Math.round(v);if(v===N)return;N=v;n=Math.min(n,N);while(w.length<N)w.push(0.5+Math.random());w=w.slice(0,N);normW();build();});
    ctrl.append('span').html('&ensp;$n$ = ');
    ctrl.append('input').attr('type','number').attr('min',0).attr('max',N).attr('value',n)
      .style('width','44px').style('font-family','inherit').style('font-size','inherit')
      .style('border','1px solid #ccc').style('border-radius','3px').style('padding','2px 4px')
      .on('change input',function(){var v=+this.value;if(isNaN(v))return;if(v<0||v>N){d3.select('#cps-status').text('the distribution requires 0 \u2264 n \u2264 N='+N).style('color','#c00');return;}v=Math.round(v);if(v===n)return;n=v;build();});

    // Shared column widths for all three tables
    var colLabelW = 'auto';
    var colItemW = (bw + 8) + 'px';
    var colPSW = '55px';
    function addCols(table) {
      var cg = table.append('colgroup');
      cg.append('col').style('width', colLabelW);
      for (var j=0;j<N;j++) cg.append('col').style('width', colItemW);
      cg.append('col').style('width', colPSW);
    }

    // === WEIGHTS TABLE (with bars) ===
    var wt = root.append('table');
    addCols(wt);
    // Header
    var wh = wt.append('thead').append('tr');
    wh.append('th').attr('class','rl').text('weights');
    for(var i=0;i<N;i++) wh.append('th').attr('class','ic').style('color',CW).html('$w_'+(i+1)+'$');
    wh.append('th').attr('class','pc');
    // Bar row
    var wb = wt.append('tbody').append('tr');
    wb.append('td').attr('class','rl').style('font-size','0.75em').style('color','#999').html('drag to adjust<br>(normalized: $\\sum w_i = 1$)');
    for(var i=0;i<N;i++){
      (function(idx){
        var td = wb.append('td').attr('class','bar-td');
        var b = makeBar(td, w[idx], getMaxW(), CW, true, function(frac){
          var target = Math.max(0, Math.min(0.99, frac));
          var EPS = 0.005;
          // Count how many items would remain with meaningful weight
          var wouldBeNonzero = 0;
          for(var k=0;k<N;k++) {
            if(k===idx) { if(target>EPS) wouldBeNonzero++; }
            else { if(w[k]>EPS) wouldBeNonzero++; }
          }
          if(wouldBeNonzero < n) {
            d3.select('#cps-status')
              .text('the distribution requires at least n='+n+' items with positive weight')
              .style('color','#c00');
            return;
          }
          d3.select('#cps-status').text('');
          // Scale the *others* to keep sum = 1
          var others = 0;
          for(var k=0;k<N;k++) if(k!==idx) others+=w[k];
          var remain = 1 - target;
          if(others > 1e-20 && remain > 0) {
            var scale = remain / others;
            for(var k=0;k<N;k++){
              if(k===idx) w[k]=target;
              else w[k]*=scale;
            }
          } else {
            w[idx] = target;
          }
          normW();
          pi = getPi(w);
          update();
        });
        wFills.push(b.fill); wLabels.push(b.label);
      })(i);
    }
    wb.append('td').attr('class','pc');

    // === SUBSETS TABLE ===
    var allS = subs(N,n);
    probCells = [];
    if (allS.length <= 70) {
      var tdata = getTable(w);
      var st = root.append('table');
      addCols(st);
      var sh = st.append('thead').append('tr');
      sh.append('th').attr('class','rl').html('Subset $S$');
      for(var j=0;j<N;j++) sh.append('th').attr('class','ic').text(j+1);
      sh.append('th').attr('class','pc').html('$P(S)$');
      var sb = st.append('tbody');
      tdata.forEach(function(r){
        var tr = sb.append('tr');
        tr.append('td').attr('class','rl').text('{'+r.s.map(function(j){return j+1;}).join(', ')+'}');
        r.ind.forEach(function(v){tr.append('td').attr('class','ic'+(v?'':' zero')).text(v);});
        var pc = tr.append('td').attr('class','pc').style('position','relative');
        pc.append('div').attr('class','prob-bar')
          .style('position','absolute').style('top','2px').style('bottom','2px').style('left','0')
          .style('background',CP).style('opacity','0.12').style('border-radius','2px')
          .style('pointer-events','none');
        pc.append('span').style('position','relative');
        probCells.push(pc);
      });
    } else {
      root.append('div').style('color','#999').style('font-size','0.85em').text('('+allS.length+' subsets\u2014table hidden)');
    }

    // === INCLUSION PROBABILITIES TABLE (with bars, draggable → fits w) ===
    // Note: dragging π_i sets a target and solves for w. Because π must sum
    // to n, changing one π_i will cause the others to adjust.
    var pt = root.append('table');
    addCols(pt);
    var ph = pt.append('thead').append('tr').style('border-top','1px solid #ccc');
    ph.append('th').attr('class','rl').style('font-weight','bold').html('inclusion prob.');
    for(var i=0;i<N;i++) ph.append('th').attr('class','ic').style('color',CP).style('font-weight','bold').html('$\\pi_'+(i+1)+'$');
    ph.append('th').attr('class','pc');
    // Bar row
    var pb = pt.append('tbody').append('tr');
    pb.append('td').attr('class','rl').style('font-size','0.75em').style('color','#999').html('drag to set target<br>(others adjust since $\\sum \\pi_i = n$)');
    for(var i=0;i<N;i++){
      (function(idx){
        var td = pb.append('td').attr('class','bar-td');
        var b = makeBar(td, pi[idx], 1, CP, true, function(frac){
          // Set this pi as target, solve for w
          pi[idx] = Math.max(0.02, Math.min(0.98, frac));
          w = fit(pi);
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
        probCells[i].select('span').text(r.p.toFixed(3));
        probCells[i].select('.prob-bar').style('width',(r.p/maxP*100)+'%');
      });
    }
  }

  build();
})();
</script>

</div>

{% notebook conditional-poisson-sampling.ipynb cells[5:] %}
