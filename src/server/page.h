// sdxl_lora_server 웹 화면 (HTML 한 장, 외부 라이브러리 없음). server.cpp가 "/"로 내보낸다.
#pragma once

static const char * PAGE = R"HTML(<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SDXL LoRA (Vulkan)</title>
<style>
:root{--bg:#121417;--card:#1b1f24;--line:#2c323a;--fg:#e6e8eb;--dim:#9aa3ad;--acc:#5aa9ff;--ok:#4cc38a;--bad:#ff6b6b;--warn:#f0b44c}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--fg)}
header{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:2}
header h1{font-size:16px;margin:0 12px 0 0}main{padding:16px;display:grid;gap:16px;max-width:1300px;margin:auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}
h2{font-size:15px;margin:0 0 10px}button,select,input,textarea{font:inherit;color:var(--fg);background:#0f1215;border:1px solid var(--line);border-radius:6px;padding:5px 9px}
button{cursor:pointer;background:#232932}button:hover{border-color:var(--acc)}button.danger{border-color:#6a2c2c;color:#ffb3b3}button.primary{background:#1f4f86;border-color:#2c6bb3}
.tabs{display:flex;gap:6px;margin-bottom:10px;flex-wrap:wrap}.tabs button.on{background:#1f4f86}
form{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px 12px;align-items:end}
form label{display:flex;flex-direction:column;gap:3px;color:var(--dim);font-size:12px}form .wide{grid-column:1/-1}form textarea{min-height:52px}
table{width:100%;border-collapse:collapse}td,th{padding:6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}th{color:var(--dim);font-weight:500;font-size:12px}
tr.sel{background:#22303f}.st{font-weight:600}.running{color:var(--acc)}.done{color:var(--ok)}.failed,.interrupted{color:var(--bad)}.cancelled,.stopped{color:var(--warn)}.queued{color:var(--dim)}
.bar{height:6px;background:#0f1215;border-radius:3px;overflow:hidden;min-width:90px}.bar i{display:block;height:100%;background:var(--acc)}
.small{font-size:12px;color:var(--dim)}pre{background:#0f1215;border:1px solid var(--line);border-radius:6px;padding:8px;max-height:320px;overflow:auto;font-size:12px;white-space:pre-wrap;word-break:break-all}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px}.grid figure{margin:0}.grid img{width:100%;border-radius:6px;display:block}
.grid figcaption{font-size:11px;color:var(--dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}a{color:var(--acc)}.row{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
#msg{min-height:18px}.err{color:var(--bad)}
</style></head><body>
<div id="gate" hidden style="position:fixed;inset:0;background:var(--bg);z-index:9;display:flex;align-items:center;justify-content:center">
<form id="gateF" class="card" style="display:flex;flex-direction:column;gap:10px;min-width:300px"><h2 style="margin:0">SDXL LoRA · Vulkan</h2>
<span class="small" id="gateMsg"></span><input id="gateIn" type="password" autocomplete="current-password" autofocus><button class="primary" id="gateBtn"></button><span id="gateErr" class="small err"></span></form></div>
<header><h1>SDXL LoRA · Vulkan</h1>
<span id="qstate" class="small"></span>
<button id="bPause"></button><button id="bCancelAll" class="danger"></button><button id="bKillRun" class="danger"></button>
<span style="flex:1"></span><button id="bLang">EN</button></header>
<main>
<section class="card"><h2 data-t="add"></h2>
<div class="tabs" id="tabs"></div><div id="forms"></div><div id="msg" class="small"></div></section>
<section class="card"><h2 data-t="queue"></h2><table><thead><tr><th>#</th><th data-t="type"></th><th data-t="what"></th><th data-t="state"></th><th data-t="progress"></th><th></th></tr></thead><tbody id="jobs"></tbody></table></section>
<section class="card" id="detail" hidden><h2 id="dTitle"></h2><div id="dOut" class="row"></div><pre id="dLog"></pre></section>
<section class="card"><h2 data-t="results"></h2><div id="results"></div></section>
<section class="card"><h2 data-t="gallery"></h2><div id="gallery" class="grid"></div></section>
</main>
<script>
const T={ko:{add:"작업 추가",queue:"큐",type:"종류",what:"내용",state:"상태",progress:"진행",results:"끝난 작업 (결과 보관함)",selAll:"전부 선택",selNone:"선택 해제",delSel:"선택 삭제",delFiles:"결과 파일도 함께 삭제 (LoRA·그림·로그)",confirmDel:"선택한 작업 {n}개를 목록에서 지울까요?",confirmDelFiles:"선택한 작업 {n}개와 그 결과 파일(LoRA·그림·로그·캐시)을 영구 삭제할까요? 되돌릴 수 없습니다.",deleted:"지움",gallery:"생성 그림",
cache:"캐시 만들기",train:"학습",generate:"생성",pipe:"한 번에 (사진→학습→생성)",pause:"큐 일시정지",resume:"큐 재개",cancelAll:"대기 전부 취소",killRun:"실행 중 강제 종료",
paused:"일시정지됨 — 새 작업을 시작하지 않음",running:"실행 중",idle:"대기 없음",nq:"대기",submit:"큐에 넣기",cancel:"취소",kill:"강제 종료",up:"▲",down:"▼",remove:"목록에서 지우기",log:"로그",
src:"이미지 폴더",size:"크기(px)",caption:"캡션 (같은 이름 .txt가 있으면 그것)",prefix:"캡션 앞에 붙일 말 (트리거)",dst:"캐시 폴더 (비우면 자동)",vaecpu:"VAE를 CPU로 (1024px 권장)",
data:"캐시 폴더",name:"이름",out:"LoRA 파일 (비우면 out/이름.safetensors)",steps:"스텝",lr:"학습률",rank:"rank",alpha:"alpha",convrank:"conv rank (LoCon, 0=끔)",seed:"seed",saveevery:"중간 저장 간격 (0=끔)",
fp16:"fp16 + 손실 스케일링 (빠름)",ckpt:"그래디언트 체크포인팅 (768px 이상 필수)",prompt:"프롬프트",negative:"네거티브 프롬프트",lora:"LoRA",none:"(없음)",weight:"LoRA 가중치",width:"너비",height:"높이",cfg:"cfg",count:"장 수 (seed를 1씩 늘림)",
prompts:"미리보기 프롬프트 (한 줄에 하나)",gsize:"미리보기 크기",added:"큐에 넣음",confirmKill:"실행 중인 작업을 즉시 강제 종료할까요?",confirmCancelAll:"대기 중인 작업을 모두 취소할까요?",
token:"접속 토큰을 입력하세요 (서버를 켠 콘솔에 출력됨)",enter:"들어가기",badToken:"토큰이 맞지 않음",download:"내려받기",openimg:"원본",nores:"아직 끝난 작업이 없음",eta:"남음",min:"분",sec:"초",cachedir:"캐시",loraf:"LoRA",image:"그림",
qtip:"작업은 한 번에 하나씩 차례로 실행됩니다 (GPU 메모리 때문). 끝난 결과는 아래 보관함에 남습니다.",nosd:"서버에 --sd-cli가 없어 캐시·생성 작업은 쓸 수 없음"},
en:{add:"Add job",queue:"Queue",type:"Type",what:"Job",state:"State",progress:"Progress",results:"Finished jobs (results)",selAll:"Select all",selNone:"Select none",delSel:"Delete selected",delFiles:"Also delete result files (LoRA, images, logs)",confirmDel:"Remove {n} selected jobs from the list?",confirmDelFiles:"Permanently delete {n} selected jobs and their result files (LoRA, images, logs, cache)? This cannot be undone.",deleted:"Deleted",gallery:"Generated images",
cache:"Build cache",train:"Train",generate:"Generate",pipe:"All in one (photos→train→generate)",pause:"Pause queue",resume:"Resume queue",cancelAll:"Cancel all queued",killRun:"Kill running job",
paused:"Paused — no new job will start",running:"running",idle:"queue empty",nq:"queued",submit:"Add to queue",cancel:"Cancel",kill:"Kill",up:"▲",down:"▼",remove:"Remove from list",log:"Log",
src:"Image folder",size:"Size (px)",caption:"Caption (a .txt with the same name wins)",prefix:"Caption prefix (trigger)",dst:"Cache folder (auto if empty)",vaecpu:"VAE on CPU (recommended at 1024px)",
data:"Cache folder",name:"Name",out:"LoRA file (default out/<name>.safetensors)",steps:"Steps",lr:"Learning rate",rank:"rank",alpha:"alpha",convrank:"conv rank (LoCon, 0=off)",seed:"seed",saveevery:"Save every N steps (0=off)",
fp16:"fp16 + loss scaling (faster)",ckpt:"Gradient checkpointing (required at 768px+)",prompt:"Prompt",negative:"Negative prompt",lora:"LoRA",none:"(none)",weight:"LoRA weight",width:"Width",height:"Height",cfg:"cfg",count:"Images (seed +1 each)",
prompts:"Preview prompts (one per line)",gsize:"Preview size",added:"Added",confirmKill:"Kill the running job now?",confirmCancelAll:"Cancel all queued jobs?",
token:"Enter the access token (printed on the server console)",enter:"Enter",badToken:"Wrong token",download:"Download",openimg:"Full size",nores:"No finished jobs yet",eta:"left",min:"min",sec:"s",cachedir:"cache",loraf:"LoRA",image:"image",
qtip:"Jobs run one at a time (GPU memory). Finished results stay in the list below.",nosd:"The server has no --sd-cli, so cache and generate jobs are unavailable"}};
let L=(localStorage.lang||(navigator.language||"ko").slice(0,2))==="ko"?"ko":"en";const t=k=>T[L][k]||k;
const $=s=>document.querySelector(s);const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
let CFG={},LIST={loras:[],caches:[],image_dirs:[]},JOBS=[],SEL=null,TAB="pipe",PAUSED=false;
function setTok(k){document.cookie="sdxl_token="+encodeURIComponent(k)+";path=/;SameSite=Strict";}
let GATED=false;
function gate(err){if(GATED&&!err)return new Promise(()=>{});GATED=true;$("#gate").hidden=false;$("#gate").style.display="flex";$("#gateMsg").textContent=t("token");$("#gateBtn").textContent=t("enter");
 $("#gateErr").textContent=err?t("badToken"):"";$("#gateIn").focus();
 $("#gateF").onsubmit=async e=>{e.preventDefault();setTok($("#gateIn").value.trim());const r=await fetch("/api/config");if(r.ok)location.reload();else gate(true);};
 return new Promise(()=>{});}  // 토큰을 넣기 전에는 아무것도 진행하지 않음
async function api(path,body){const o=body===undefined?{}:{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)};
 const r=await fetch(path,o);if(r.status===401)return gate(false);
 const j=await r.json().catch(()=>({}));if(!r.ok)throw new Error(j.error||r.status);return j;}
function msg(s,err){const m=$("#msg");m.textContent=s;m.className="small"+(err?" err":"");}
const FIELDS={
 cache:[["src","dirs:image_dirs",""],["size","num",512],["caption","text","a photo of sks dog"],["prefix","text",""],["dst","text",""],["vae_backend","chk:vaecpu",false]],
 train:[["data","dirs:caches",""],["name","text",""],["out","text",""],["steps","num",200],["lr","num",0.0001],["rank","num",8],["alpha","num",8],["conv_rank","num",0],["seed","num",42],["save_every","num",0],["fp16","chk:fp16",true],["ckpt","chk:ckpt",true]],
 generate:[["prompt","area","a photo of sks dog on a sunny beach"],["negative","text",""],["lora","files:loras",""],["lora_weight","num:weight",1],["width","num",1024],["height","num",1024],["steps","num",20],["cfg","num",7],["seed","num",42],["count","num",1]],
 pipe:[["src","dirs:image_dirs",""],["name","text","mylora"],["size","num",512],["caption","text","a photo of sks dog"],["steps","num",200],["conv_rank","num",0],["prompts","area","a photo of sks dog on a sunny beach\na photo of sks dog in a bucket"],["gsize","num",1024],["vae_backend","chk:vaecpu",false]]};
const LBL={lora_weight:"weight",conv_rank:"convrank",save_every:"saveevery"};
function renderForms(){
 $("#tabs").innerHTML=["pipe","cache","train","generate"].map(k=>`<button data-k="${k}" class="${k===TAB?"on":""}">${t(k)}</button>`).join("");
 $("#tabs").querySelectorAll("button").forEach(b=>b.onclick=()=>{TAB=b.dataset.k;renderForms();});
 const f=FIELDS[TAB];let h=`<form id="F">`;
 for(const [k,kind,def] of f){const [ty,sub]=kind.split(":");const lab=t(sub&&ty!=="dirs"&&ty!=="files"?sub:(LBL[k]||k));
  const v=localStorage["f_"+TAB+"_"+k]??def;
  if(ty==="num")h+=`<label>${lab}<input name="${k}" type="number" step="any" value="${esc(v)}"></label>`;
  else if(ty==="text")h+=`<label>${lab}<input name="${k}" value="${esc(v)}"></label>`;
  else if(ty==="area")h+=`<label class="wide">${lab}<textarea name="${k}">${esc(v)}</textarea></label>`;
  else if(ty==="chk")h+=`<label><span><input name="${k}" type="checkbox" ${String(v)==="true"?"checked":""}> ${lab}</span></label>`;
  else{const opts=LIST[sub]||[];h+=`<label>${lab}<input name="${k}" list="dl_${k}" value="${esc(v)}"><datalist id="dl_${k}">${opts.map(o=>`<option value="${esc(o)}">`).join("")}</datalist></label>`;}}
 h+=`<div class="row"><button class="primary" type="submit">${t("submit")}</button></div></form>`;
 if(!CFG.sd_cli&&TAB!=="train")h+=`<div class="small err">${t("nosd")}</div>`;
 $("#forms").innerHTML=h;$("#F").onsubmit=e=>{e.preventDefault();submit();};}
function formVals(){const F=$("#F"),o={};for(const [k,kind] of FIELDS[TAB]){const el=F.elements[k];let v;
  if(kind.startsWith("chk"))v=el.checked;else if(kind.startsWith("num"))v=el.value===""?undefined:Number(el.value);else v=el.value.trim();
  localStorage["f_"+TAB+"_"+k]=v;if(v!==undefined&&v!=="")o[k]=v;}return o;}
async function submit(){try{const v=formVals();let ids=[];
 if(TAB==="cache"){if(v.vae_backend)v.vae_backend="cpu";else delete v.vae_backend;ids.push((await api("/api/jobs",{type:"cache",...v})).id);}
 else if(TAB==="train"){ids.push((await api("/api/jobs",{type:"train",...v})).id);}
 else if(TAB==="generate"){const n=Math.max(1,Math.min(50,v.count||1));delete v.count;for(let i=0;i<n;i++)ids.push((await api("/api/jobs",{type:"generate",...v,seed:(v.seed||0)+i})).id);}
 else{const nm=(v.name||"mylora").replace(/[^A-Za-z0-9._-]/g,"_"),size=v.size||512,cache=`cache/${nm}_${size}`,lora=`out/${nm}.safetensors`;
  ids.push((await api("/api/jobs",{type:"cache",src:v.src,dst:cache,size,caption:v.caption||"",...(v.vae_backend?{vae_backend:"cpu"}:{})})).id);
  ids.push((await api("/api/jobs",{type:"train",data:cache,out:lora,steps:v.steps||200,conv_rank:v.conv_rank||0,ckpt:size>=768||true})).id);
  for(const p of String(v.prompts||"").split("\n").map(s=>s.trim()).filter(Boolean))ids.push((await api("/api/jobs",{type:"generate",prompt:p,lora,width:v.gsize||1024,height:v.gsize||1024})).id);}
 msg(t("added")+": #"+ids.join(", #"));refresh();}catch(e){msg(String(e.message||e),true);}}
const fmtT=s=>s>=60?Math.round(s/60)+t("min"):Math.round(s)+t("sec");
function what(j){const p=j.params||{};if(j.type==="train")return `${p.data||""} → ${p.out||("out/"+(p.name||"")+".safetensors")} · ${p.steps??200} steps`;
 if(j.type==="generate")return `${p.prompt||""}${p.lora?" · "+p.lora.split("/").pop():""} · ${p.width??1024}×${p.height??1024} · seed ${p.seed??42}`;
 return `${p.src||""} → ${p.dst||"cache/…"} · ${p.size??512}px`;}
function prog(j){const g=j.progress;if(!g)return j.state==="running"?`<span class="small">…</span>`:"";const pct=Math.round(100*g.step/g.steps);
 return `<div class="bar"><i style="width:${pct}%"></i></div><span class="small">${g.step}/${g.steps}${g.loss?" · loss "+esc(g.loss):""}${g.eta_sec!==undefined?" · "+fmtT(g.eta_sec)+" "+t("eta"):""}</span>`;}
function renderJobs(){const tb=$("#jobs");
 tb.innerHTML=JOBS.slice().sort((a,b)=>{const r=s=>s==="running"?0:s==="queued"?1:2;return r(a.state)-r(b.state)||(r(a.state)===2?b.id-a.id:0);}).map(j=>{
  let act=`<button data-a="log" data-id="${j.id}">${t("log")}</button>`;
  if(j.state==="queued")act+=`<button data-a="up" data-id="${j.id}">${t("up")}</button><button data-a="down" data-id="${j.id}">${t("down")}</button><button data-a="cancel" data-id="${j.id}" class="danger">${t("cancel")}</button>`;
  else if(j.state==="running")act+=`<button data-a="cancel" data-id="${j.id}" class="danger">${t("cancel")}</button><button data-a="kill" data-id="${j.id}" class="danger">${t("kill")}</button>`;
  else act+=`<button data-a="remove" data-id="${j.id}">✕</button>`;
  return `<tr class="${SEL===j.id?"sel":""}"><td>${j.id}</td><td>${t(j.type)}</td><td>${esc(what(j))}${j.error?`<div class="small err">${esc(j.error)}</div>`:""}</td><td class="st ${j.state}">${j.state}</td><td>${prog(j)}</td><td class="row">${act}</td></tr>`;}).join("")
  ||`<tr><td colspan=6 class="small">${t("qtip")}</td></tr>`;
 tb.querySelectorAll("button").forEach(b=>b.onclick=()=>act(b.dataset.a,+b.dataset.id));}
async function act(a,id){try{
 if(a==="log"){SEL=id;await showDetail();renderJobs();return;}
 if(a==="kill"&&!confirm(t("confirmKill")))return;
 if(a==="up"||a==="down")await api(`/api/jobs/${id}/move`,{dir:a});else await api(`/api/jobs/${id}/${a}`,{});
 if(a==="remove"&&SEL===id){SEL=null;$("#detail").hidden=true;}refresh();}catch(e){msg(String(e.message||e),true);}}
function outLinks(j){return (j.outputs||[]).map(o=>{const u="/files/"+o.path.split("/").map(encodeURIComponent).join("/");
 if(o.kind==="image")return `<a href="${u}" target="_blank"><img src="${u}" style="height:72px;border-radius:4px"></a><a href="${u}" download>${t("download")}</a>`;
 if(o.kind==="lora")return `<a href="${u}" download>⬇ ${esc(o.path.split("/").pop())}</a>`;return `<span class="small">${t("cachedir")}: ${esc(o.path)}</span>`;}).join(" ");}
async function showDetail(){if(SEL==null)return;const j=await api("/api/jobs/"+SEL);$("#detail").hidden=false;
 $("#dTitle").innerHTML=`#${j.id} ${t(j.type)} · <span class="st ${j.state}">${j.state}</span> · <a href="/api/jobs/${j.id}/log" target="_blank">${t("log")}</a>`;
 $("#dOut").innerHTML=outLinks(j);const pre=$("#dLog"),bottom=pre.scrollTop+pre.clientHeight>=pre.scrollHeight-4;pre.textContent=(j.log_tail||[]).join("\n");if(bottom)pre.scrollTop=pre.scrollHeight;}
let CHK=new Set();
function renderResults(){const fin=JOBS.filter(j=>j.state!=="queued"&&j.state!=="running").sort((a,b)=>b.id-a.id);
 for(const id of [...CHK])if(!fin.some(j=>j.id===id))CHK.delete(id);
 $("#results").innerHTML=fin.length?`<div class="row" style="margin-bottom:8px"><button id="rAll">${t("selAll")}</button><button id="rNone">${t("selNone")}</button>
  <label class="small"><input type="checkbox" id="rFiles"> ${t("delFiles")}</label><button id="rDel" class="danger">${t("delSel")} (${CHK.size})</button></div>
  <table><tbody>${fin.map(j=>`<tr><td><input type="checkbox" class="rc" data-id="${j.id}" ${CHK.has(j.id)?"checked":""}></td><td>#${j.id}</td><td>${t(j.type)}</td><td class="st ${j.state}">${j.state}</td><td>${esc(what(j))}${j.error?`<div class="small err">${esc(j.error)}</div>`:""}</td><td class="row">${outLinks(j)} <a href="/api/jobs/${j.id}/log" target="_blank" download="job${j.id}.log.txt">${t("log")}</a></td></tr>`).join("")}</tbody></table>`
  :`<span class="small">${t("nores")}</span>`;
 if(fin.length){const upd=()=>{$("#rDel").textContent=`${t("delSel")} (${CHK.size})`;};
  document.querySelectorAll(".rc").forEach(c=>c.onchange=()=>{c.checked?CHK.add(+c.dataset.id):CHK.delete(+c.dataset.id);upd();});
  $("#rAll").onclick=()=>{fin.forEach(j=>CHK.add(j.id));renderResults();};$("#rNone").onclick=()=>{CHK.clear();renderResults();};
  $("#rDel").onclick=async()=>{if(!CHK.size)return;const files=$("#rFiles").checked;
   if(!confirm(t(files?"confirmDelFiles":"confirmDel").replace("{n}",CHK.size)))return;
   try{const r=await api("/api/jobs/remove-many",{ids:[...CHK],delete_files:files});msg(`${t("deleted")}: ${r.removed}`+(files?` (files ${r.files_deleted})`:""));
    if(CHK.has(SEL)){SEL=null;$("#detail").hidden=true;}CHK.clear();lastDone="";refresh();}catch(e){msg(String(e.message||e),true);}};}
 const imgs=fin.filter(j=>j.type==="generate"&&j.state==="done").flatMap(j=>(j.outputs||[]).filter(o=>o.kind==="image").map(o=>({j,o})));
 $("#gallery").innerHTML=imgs.map(({j,o})=>{const u="/files/"+o.path.split("/").map(encodeURIComponent).join("/");
  return `<figure><a href="${u}" target="_blank"><img src="${u}" loading="lazy"></a><figcaption title="${esc(j.params.prompt)}">#${j.id} ${esc(j.params.prompt)}</figcaption></figure>`;}).join("");}
function renderHeader(){const run=JOBS.find(j=>j.state==="running"),nq=JOBS.filter(j=>j.state==="queued").length;
 $("#qstate").innerHTML=(PAUSED?`<span class="err">${t("paused")}</span> · `:"")+(run?`${t("running")}: #${run.id} ${t(run.type)} · `:"")+`${t("nq")} ${nq}`;
 $("#bPause").textContent=PAUSED?t("resume"):t("pause");$("#bCancelAll").textContent=t("cancelAll");$("#bKillRun").textContent=t("killRun");$("#bKillRun").disabled=!run;
 document.querySelectorAll("[data-t]").forEach(e=>e.textContent=t(e.dataset.t));$("#bLang").textContent=L==="ko"?"EN":"한국어";}
let lastDone="";
async function refresh(){try{const r=await api("/api/jobs");JOBS=r.jobs;PAUSED=r.paused;renderHeader();renderJobs();
 const d=JOBS.filter(j=>j.state!=="queued"&&j.state!=="running").map(j=>j.id+j.state).join();if(d!==lastDone){lastDone=d;renderResults();LIST=await api("/api/list");}
 if(SEL!=null)await showDetail();}catch(e){msg(String(e.message||e),true);}}
$("#bPause").onclick=()=>api("/api/queue/"+(PAUSED?"resume":"pause"),{}).then(refresh);
$("#bCancelAll").onclick=()=>confirm(t("confirmCancelAll"))&&api("/api/queue/cancel-all",{}).then(refresh);
$("#bKillRun").onclick=()=>{const run=JOBS.find(j=>j.state==="running");if(run&&confirm(t("confirmKill")))api(`/api/jobs/${run.id}/kill`,{}).then(refresh);};
$("#bLang").onclick=()=>{L=L==="ko"?"en":"ko";localStorage.lang=L;renderHeader();renderForms();renderJobs();renderResults();};
(async()=>{const qs=new URLSearchParams(location.search);if(qs.get("token")){setTok(qs.get("token"));history.replaceState(null,"",location.pathname);}
 try{CFG=await api("/api/config");LIST=await api("/api/list");}catch(e){msg(String(e.message||e),true);}renderForms();await refresh();setInterval(refresh,3000);})();
</script></body></html>)HTML";
