const form=document.querySelector('#form'),run=document.querySelector('#run'),badge=document.querySelector('#badge'),empty=document.querySelector('#empty'),content=document.querySelector('#content'),wrap=document.querySelector('#progressWrap'),bar=document.querySelector('#bar'),progressText=document.querySelector('#progressText');
const rounding={parent_quantity:3,duration_seconds:0,volatility:2,spread:2,arrival_rate:3,displayed_depth:2,seed:0};

form.addEventListener('submit',async event=>{
  event.preventDefault();run.disabled=true;empty.classList.add('hidden');content.classList.add('hidden');wrap.classList.remove('hidden');badge.textContent='Running';bar.style.background='';
  const data=Object.fromEntries(new FormData(form));
  Object.entries(rounding).forEach(([key,digits])=>{data[key]=Number(Number(data[key]).toFixed(digits));const input=form.elements[key];if(input)input.value=data[key].toFixed(digits)});
  try{const response=await fetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});if(!response.ok)throw new Error(await response.text());poll((await response.json()).id)}catch(error){fail(error.message)}
});

async function poll(id){try{const response=await fetch(`/api/jobs/${id}`),job=await response.json();bar.style.width=`${job.progress||0}%`;progressText.textContent=job.message||job.status;if(job.status==='complete')return render(job.result);if(job.status==='failed')return fail(job.message);setTimeout(()=>poll(id),400)}catch(error){fail(error.message)}}
function fail(message){run.disabled=false;badge.textContent='Error';progressText.textContent=message;bar.style.width='100%';bar.style.background='#a63232'}
function metric(label,value){return `<div class="metric"><b>${value}</b><span>${label}</span></div>`}

function render(result){
  run.disabled=false;wrap.classList.add('hidden');content.classList.remove('hidden');badge.textContent='Complete';const summary=result.summary;
  document.querySelector('#metrics').innerHTML=metric('Adaptive completion',`${(summary.completion*100).toFixed(2)}%`)+metric('Adaptive cost',`${summary.penalized_cost_bp.toFixed(4)} bp`)+metric('Average fill',summary.average_fill_price?`$${summary.average_fill_price.toFixed(2)}`:'—')+metric('Child instructions',summary.instructions);
  renderComparison(result.comparison);draw(result.execution_path,result.request.duration_seconds);
  document.querySelector('#rows').innerHTML=result.execution_path.map(row=>`<tr><td>${row.elapsed_seconds.toFixed(1)}s</td><td>${row.requested.toFixed(5)}</td><td>${row.filled.toFixed(5)}</td><td>${row.cumulative_filled.toFixed(5)}</td><td>${row.remaining.toFixed(5)}</td><td>${row.causal_impact_bp==null?'—':row.causal_impact_bp.toFixed(4)+' bp'}</td></tr>`).join('');
}

function renderComparison(comparison){
  const items=['immediate','twap','adaptive'].map(key=>comparison[key]),maxImpact=Math.max(...items.map(x=>x.impact_bp),1e-9),maxCost=Math.max(...items.map(x=>x.penalized_cost_bp),1e-9);
  document.querySelector('#comparison').innerHTML=items.map(item=>`<article class="compare-card ${item.label==='Adaptive'?'recommended':''}"><div class="compare-title"><b>${item.label}</b>${item.label==='Adaptive'?'<span>Recommended</span>':''}</div><div class="compare-value"><strong>${item.impact_bp.toFixed(4)}</strong> bp impact</div><div class="mini-track"><i style="width:${Math.max(2,100*item.impact_bp/maxImpact)}%"></i></div><div class="compare-value"><strong>${item.penalized_cost_bp.toFixed(4)}</strong> bp cost</div><div class="mini-track cost"><i style="width:${Math.max(2,100*item.penalized_cost_bp/maxCost)}%"></i></div><small>${(item.completion*100).toFixed(1)}% completed</small></article>`).join('');
}

function draw(path,duration){
  const svg=document.querySelector('#chart'),W=900,H=390,pL=62,pR=24,pT=22,pB=54,x=t=>pL+(W-pL-pR)*t/duration,y=v=>H-pB-(H-pT-pB)*v;
  const actual=[[0,0],...path.map(row=>[row.elapsed_seconds,row.executed_fraction])],line=actual.map((point,index)=>`${index?'L':'M'}${x(point[0]).toFixed(1)},${y(point[1]).toFixed(1)}`).join(' ');let marks='';
  for(let i=0;i<=4;i++){const time=duration*i/4,xx=x(time),fraction=i/4,yy=y(fraction);marks+=`<line class="tick" x1="${xx}" y1="${H-pB}" x2="${xx}" y2="${H-pB+6}"/><text class="tick-label" x="${xx}" y="${H-pB+23}" text-anchor="middle">${Math.round(time)}s</text><line class="tick" x1="${pL-6}" y1="${yy}" x2="${pL}" y2="${yy}"/><text class="tick-label" x="${pL-11}" y="${yy+4}" text-anchor="end">${Math.round(fraction*100)}%</text>`}
  svg.innerHTML=`<line class="axis" x1="${pL}" y1="${H-pB}" x2="${W-pR}" y2="${H-pB}"/><line class="axis" x1="${pL}" y1="${pT}" x2="${pL}" y2="${H-pB}"/>${marks}<path class="actual-line" d="${line}"/><text class="axis-title" x="${(pL+W-pR)/2}" y="${H-8}" text-anchor="middle">Elapsed execution time</text><text class="axis-title" x="17" y="${(pT+H-pB)/2}" transform="rotate(-90 17 ${(pT+H-pB)/2})" text-anchor="middle">Parent order executed</text>`;
}
