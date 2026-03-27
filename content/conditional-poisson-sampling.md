title: Conditional Poisson Sampling
date: 2026-03-25
comments: true
tags: notebook, sampling, algorithms, sampling-without-replacement

{% notebook conditional-poisson-sampling.ipynb cells[1:4] %}

<style>
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

    // === ONE TABLE for everything ===
    var tbl = root.append('table');
    var cg = tbl.append('colgroup');
    cg.append('col').style('width', 'auto');
    for (var j=0;j<N;j++) cg.append('col').style('width', (bw+8)+'px');
    cg.append('col').style('width', '55px');
    var tbody = tbl.append('tbody');

    // --- Weights header ---
    var wh = tbody.append('tr');
    wh.append('td').attr('class','rl').text('weights');
    for(var i=0;i<N;i++) wh.append('td').attr('class','ic').style('color',CW).html('$w_'+(i+1)+'$');
    wh.append('td').attr('class','pc');

    // --- Weight bars ---
    var wb = tbody.append('tr');
    wb.append('td').attr('class','rl').style('font-size','0.7em').style('color','#999').html('drag to adjust<br>($\\sum w_i = 1$)');
    for(var i=0;i<N;i++){
      (function(idx){
        var td = wb.append('td').attr('class','bar-td');
        var b = makeBar(td, w[idx], getMaxW(), CW, true, function(frac){
          var target = Math.max(0, Math.min(0.99, frac));
          var EPS = 0.005;
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

    // --- Spacer ---
    tbody.append('tr').append('td').attr('colspan',N+2).style('height','6px');

    // --- Subsets header ---
    var allS = subs(N,n);
    probCells = [];
    if (allS.length <= 70) {
      var tdata = getTable(w);
      var sh = tbody.append('tr');
      sh.append('td').attr('class','rl').style('text-align','right').html('Subset $S$');
      for(var j=0;j<N;j++) sh.append('td').attr('class','ic').text(j+1);
      sh.append('td').attr('class','pc').style('text-align','left').html('$P(S)$');

      // --- Subset rows ---
      tdata.forEach(function(r){
        var tr = tbody.append('tr');
        tr.append('td').attr('class','rl').style('color','#333').style('font-style','normal').style('text-align','right').text('{'+r.s.map(function(j){return j+1;}).join(', ')+'}');
        r.ind.forEach(function(v){tr.append('td').attr('class','ic'+(v?'':' zero')).text(v);});
        var pc = tr.append('td').attr('class','pc').style('position','relative').style('text-align','left');
        pc.append('div').attr('class','prob-bar')
          .style('position','absolute').style('top','2px').style('bottom','2px').style('left','0')
          .style('background',CP).style('opacity','0.12').style('border-radius','2px')
          .style('pointer-events','none');
        pc.append('span').style('position','relative');
        probCells.push(pc);
      });
    } else {
      tbody.append('tr').append('td').attr('colspan',N+2).style('color','#999').style('font-size','0.85em').text('('+allS.length+' subsets\u2014table hidden)');
    }

    // --- Spacer ---
    tbody.append('tr').append('td').attr('colspan',N+2).style('height','6px');

    // --- Pi header ---
    var ph = tbody.append('tr');
    ph.append('td').attr('class','rl').style('font-weight','bold').html('inclusion prob.');
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

{% notebook conditional-poisson-sampling.ipynb cells[4:12] %}

<style>
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

<div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px 20px; margin: 16px 0;">

**Interactive product tree.** Drag the weight sliders to see how changing $w_i$ affects the polynomial coefficients at every node. The tree builds the product $\prod_i(1 + w_i z)$ bottom-up; the $n$-th coefficient at the root (highlighted in red) is the normalizing constant $Z$. Use the $N$ and $n$ controls to change the problem size.

<div id="tw"></div>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
(function() {
  var N = 4, n = 2;
  var w = [0.2, 0.35, 0.15, 0.3];
  function normW() { var s=w.reduce(function(a,b){return a+b;},0); w=w.map(function(v){return v/s;}); }
  normW();

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
        w.push(0);
        N = N + 1;
        lastAction = 'add';
      } else {
        var removed = w.pop();
        N = N - 1;
        var s = w.reduce(function(a,b){return a+b;}, 0);
        if (s > 0) w = w.map(function(v) { return v / s; });
        lastAction = 'remove';
      }
      n = Math.min(n, N);
      normW();
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
    var svgW = Math.max(300, leafCount * leafSpacing + 60);

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

    var svgWrap = root.append('div')
      .style('position', 'relative')
      .style('width', svgW + 'px')
      .style('height', svgH + 'px');
    var svg = svgWrap.append('svg')
      .attr('width', svgW).attr('height', svgH)
      .style('display', 'block').style('user-select', 'none');

    // Assign x: leaves left-aligned, so adding on the right doesn't shift the left
    var leafCount = levels[0].length;
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

    // --- Draw histograms for all nodes ---
    // Find global max coefficient for consistent scaling
    var globalMax = 0;
    (function walk(nd) {
      if (!nd || nd.pad) return;
      nd.poly.forEach(function(c) { if (Math.abs(c) > globalMax) globalMax = Math.abs(c); });
      if (nd.left) walk(nd.left);
      if (nd.right) walk(nd.right);
    })(tree.root);
    if (globalMax === 0) globalMax = 1;

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
          var barFrac = Math.min(1, Math.abs(coeffs[k]) / globalMax);
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
        var frac = Math.min(w[idx], 1);
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
            var target = Math.max(0.01, Math.min(0.99, frac));
            var others = 0;
            for (var k = 0; k < N; k++) if (k !== idx) others += w[k];
            var remain = 1 - target;
            if (others > 1e-10 && remain > 0) {
              var scale = remain / others;
              for (var k = 0; k < N; k++) {
                if (k === idx) w[k] = target;
                else w[k] = Math.max(0.001, w[k] * scale);
              }
            }
            normW();
            updateTree();
          }));
      })(idx);
    });

    // Z label in output zone, centered on the n-th bar
    zLabelDiv = addMathLabel(svgWrap, zLabelX, outputY, '', {anchor:'middle', color: CR, fontSize: '14px'});

    root.datum({ levels: levels, tree: tree, globalMax: globalMax });
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
      var frac = Math.min(w[i], 1);
      sliderFills[i].attr('y', sliderTop + sliderH - frac * sliderH).attr('height', frac * sliderH);
      sliderLabels[i].attr('y', sliderTop + sliderH - frac * sliderH - 2).text(w[i].toFixed(2));
    }

    // Rebuild tree data
    var newTree = buildTree(w, n);
    var newLevels = newTree.levels;

    // Recompute global max
    var gMax = 0;
    (function walk(nd) {
      if (!nd || nd.pad) return;
      nd.poly.forEach(function(c) { if (Math.abs(c) > gMax) gMax = Math.abs(c); });
      if (nd.left) walk(nd.left);
      if (nd.right) walk(nd.right);
    })(newTree.root);
    if (gMax === 0) gMax = 1;

    // Update histograms
    var ri = 0;
    for (var li = 0; li < newLevels.length; li++) {
      for (var ni = 0; ni < newLevels[li].length; ni++) {
        var ref = nodeRefs[ri++];
        if (!ref) continue;
        var coeffs = newLevels[li][ni].poly;
        var nCoeffs = ref.nCoeffs;
        for (var k = 0; k < nCoeffs; k++) {
          var barFrac = Math.min(1, Math.abs(coeffs[k]) / gMax);
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

{% notebook conditional-poisson-sampling.ipynb cells[12:] %}
