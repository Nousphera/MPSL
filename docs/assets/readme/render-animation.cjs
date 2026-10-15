/**
 * Render the existing MPSL method walkthrough without modifying its source.
 * Requires Node.js, Playwright, a Chromium browser, and ffmpeg.
 *
 * Serve the Nousphera website checkout, then run:
 *   node render-animation.cjs --url http://127.0.0.1:60664/projects/mpsl/
 * CHROME_PATH, FFMPEG and NODE_PATH may point to local installations.
 * --preview renders the six completed scenes without exporting animation.
 * --layout desktop|mobile and --theme light|dark limit the output.
 */
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {createHash} = require('node:crypto');
const {spawnSync} = require('node:child_process');
const {chromium} = require('playwright');
const args = process.argv.slice(2);
const option = (key, fallback) => args.includes(key) ? args[args.indexOf(key) + 1] : fallback;
const url = option('--url', 'http://127.0.0.1:60664/projects/mpsl/');
const output = __dirname;
const preview = args.includes('--preview');
const layouts = option('--layout', 'all') === 'all' ? ['desktop','mobile'] : [option('--layout')];
const themes = option('--theme', 'all') === 'all' ? ['light','dark'] : [option('--theme')];
const fps = 10;
const secondsPerScene = 5;
const totalSeconds = 6 * secondsPerScene;
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'mpsl-readme-'));
const ffmpeg = process.env.FFMPEG || 'ffmpeg';
const chrome = process.env.CHROME_PATH;
const css = `
html {font-size:20px;scroll-behavior:auto!important;scrollbar-gutter:auto!important;overflow:hidden;}
body {margin:0!important;padding:0!important;background:var(--surface);overflow:hidden;}
.paper-story {margin:0;border:0;border-radius:0;width:100%;overflow:hidden;}
.story-head {padding:20px 26px 13px;}
.story-head>.px-kicker {font-size:13px;letter-spacing:.055em;color:var(--muted);max-width:none!important;}
.story-head>.px-kicker::before {content:'MPSL';color:var(--accent);font-weight:700;font-size:21px;letter-spacing:-.03em;margin-right:20px;}
.story-toggle,.story-navigation,.story-expand,.story-dialog,figcaption,.sr-only {display:none!important;}
.connected-story .story-steps {padding:0 24px 17px;gap:7px;grid-template-columns:repeat(6,minmax(0,1fr));}
.connected-story .story-steps button {height:45px;min-height:45px;padding:7px 9px;font-size:14px;border-radius:7px;}
.story-steps button>span {font-size:11px;}
.story-progress {height:3px;}
.connected-story .story-viewport {padding:0 26px 22px;}
.ms-network {padding:17px 0;gap:24px;}
.ms-endpoint {padding:10px 13px;}
.ms-endpoint strong {font-size:17px;}
.ms-endpoint small {font-size:13px;}
.ms-connection>strong {font-size:15px;}
.ms-connection>small {font-size:13px;}
.ms-wire {height:17px;}
.ms-direction-back .ms-wire {transform:none;}
.ms-direction-back .ms-wire>.ms-icon {right:auto;left:-3px;transform:rotate(180deg);}
.ms-direction-back .ms-wire>i {animation-direction:reverse!important;}
.ms-server {min-height:97px;}
.ms-scene-body {min-height:390px;gap:18px;padding-top:19px;}
.ms-stage-heading h3 {font-size:27px;margin-top:5px;}
.ms-stage-heading>.ms-eyebrow {font-size:12px;}
.ms-card>header>strong {font-size:16px;}
.ms-card>header>span:last-child:not(.ms-client-tag) {font-size:13px;}
.ms-client-grid {gap:16px;}
.ms-card {padding:16px;}
.ms-card>header {margin-bottom:14px;}
.ms-tokenize-flow {gap:13px;}
.ms-sample-text {font-size:14px;}
.ms-local-models>span {font-size:13px;min-height:48px;}
.ms-sample-image,.ms-sample-text {min-height:48px;}
.ms-tokenize-card .ms-modality>.ms-icon {display:none;}
.ms-tokens>i {width:17px;height:24px;}
.ms-tokens.ms-text>i {height:13px;}
.ms-bundle>strong {font-size:15px;}
.ms-encode-layout {grid-template-columns:minmax(0,1fr) 26px minmax(0,1.2fr) 26px minmax(0,1.2fr);gap:12px;}
.ms-encoder {min-height:258px;}
.ms-encoder>strong {font-size:20px;}
.ms-encoder>small,.ms-encoder-modalities>span {font-size:14px;}
.ms-fusion>small {font-size:13px;}
.ms-objective {gap:18px;padding:16px;}
.ms-equation {gap:18px;}
.ms-gradient-return>strong {font-size:19px;}
.ms-footnote,.ms-footnote strong,.ms-legend {font-size:14px;}
.ms-backward {max-width:470px;}
.ms-updated-models>span {font-size:15px;}
.is-enhanced [data-reveal] {transition:none!important;}
.is-enhanced [data-reveal][data-revealed='false'] {opacity:0!important;transform:none!important;}
.is-enhanced [data-reveal][data-revealed='true'] {opacity:var(--capture-opacity,1)!important;transform:translateY(var(--capture-shift,0px))!important;}
@media(max-width:650px) {
 html {font-size:18px;}
 .story-head {padding:14px 15px 11px;}
 .story-head>.px-kicker {max-width:none;font-size:10px;letter-spacing:.015em;}
 .story-head>.px-kicker::before {font-size:20px;margin-right:14px;}
 .connected-story .story-steps {padding:0 14px 12px;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;}
 .connected-story .story-steps button {font-size:12px;min-height:34px;height:34px;padding:5px 7px;gap:5px;}
 .story-steps button>span {font-size:10px;}
 .connected-story .story-viewport {padding:0 16px 17px;}
 .ms-network {gap:9px;padding:14px 0;grid-template-columns:1fr .78fr 1fr;}
 .ms-endpoint {padding:7px;gap:5px;}
 .ms-endpoint strong {font-size:13px;}
 .ms-endpoint small {font-size:11px;}
 .ms-client>.ms-icon {width:18px;height:18px;}
 .ms-server {min-height:90px;}
 .ms-connection>strong {font-size:12px;}
 .ms-connection>small {font-size:11px;}
 .ms-scene-body {min-height:460px;gap:13px;padding-top:16px;}
 .ms-stage-heading h3 {font-size:21px;margin-top:3px;}
 .ms-stage-heading>.ms-eyebrow {font-size:10px;}
 .ms-card {padding:12px;}
 .ms-card>header {margin-bottom:10px;}
 .ms-card>header>strong {font-size:13px;}
 .ms-card>header>span:last-child:not(.ms-client-tag) {font-size:11px;}
 .ms-client-grid {gap:11px;}
 .ms-local-models>span {font-size:12px;min-height:43px;}
 .ms-sample-image,.ms-sample-text {min-height:43px;}
 .ms-sample-text {font-size:12px;}
 .ms-tokens>i {width:14px;height:21px;}
 .ms-tokens.ms-text>i {height:12px;}
 .ms-encode-layout {grid-template-columns:1fr;gap:7px;}
 .ms-encode-inputs {gap:10px;}
 .ms-encode-input .ms-modality {min-height:24px;}
 .ms-encoder {min-height:112px;padding:10px 17px;gap:3px 12px;}
 .ms-encoder>strong {font-size:16px;}
 .ms-encoder>small,.ms-encoder-modalities>span {font-size:12px;}
 .ms-fusion>small {font-size:12px;}
 .ms-prediction-return {margin-top:7px;}
 .ms-footnote,.ms-footnote strong,.ms-legend {font-size:12px;}
 .ms-objective {padding:13px 9px;gap:10px;}
 .ms-equation {gap:13px;}
 .ms-gradient-return>strong {font-size:16px;}
 .ms-updated-models>span {font-size:12px;}
}
`;

async function prepare(page, theme, layout) {
  await page.route('**/assets/paper-story.js*', route => route.abort());
  await page.goto(url, {waitUntil:'networkidle'});
  await page.evaluate(({theme}) => {
    document.documentElement.dataset.theme=theme;
    const source=document.querySelector('[data-story-kind="mpsl"]');
    const style=getComputedStyle(source);
    const root=source.cloneNode(true);
    for(const variable of ['--story-secondary','--story-warn']) root.style.setProperty(variable,style.getPropertyValue(variable));
    document.body.replaceChildren(root);
    root.classList.add('is-enhanced');
    root.dataset.running='true';
    root.style.setProperty('--story-count', 6);
    root.querySelector('[data-story-controls]').hidden=false;
    [...root.querySelectorAll('[data-story-scene]')].forEach((p,i) => p.hidden=i!==0);
    window.captureRoot=root;
  }, {theme});
  await page.addStyleTag({content:css});
  await page.evaluate(() => document.fonts.ready);
  const height=await page.locator('.paper-story').evaluate(el=>Math.ceil(el.getBoundingClientRect().height));
  await page.setViewportSize({width:layout==='desktop'?960:420,height:height+(height%2)});
  await frame(page,0,4000);
  const dimensions=await page.evaluate(() => ({width:innerWidth,height:innerHeight,overflow:document.documentElement.scrollWidth>innerWidth,contentHeight:Math.ceil(window.captureRoot.getBoundingClientRect().height)}));
  if(dimensions.overflow || dimensions.contentHeight>dimensions.height) throw new Error('Clipped capture: '+JSON.stringify(dimensions));
  return dimensions;
}

async function frame(page,index,elapsed) {
  await page.evaluate(({index,elapsed,secondsPerScene,totalSeconds})=>{
    const root=window.captureRoot;
    root.dataset.active=index;
    root.querySelectorAll('[data-story-scene]').forEach((panel,i)=>panel.hidden=i!==index);
    root.querySelectorAll('[data-story-step]').forEach((step,i)=>step.setAttribute('aria-pressed',String(i===index)));
    const panel=root.querySelector(`[data-story-scene="${index}"]`);
    const originalDuration=Number(panel.dataset.storyDuration);
    // Compress reveals into the first 58% of each five-second scene; hold the
    // complete diagram for reading. No method content or step order changes.
    const revealScale=secondsPerScene*1000/originalDuration;
    panel.querySelectorAll('[data-reveal]').forEach(part=>{
      const begin=Number(part.dataset.reveal)*revealScale;
      const alpha=begin===0?1:Math.max(0,Math.min(1,(elapsed-begin)/220));
      part.dataset.revealed=String(elapsed>=begin);
      part.style.setProperty('--capture-opacity',alpha);
      part.style.setProperty('--capture-shift',`${(1-alpha)*5}px`);
    });
    root.querySelector('[data-story-progress]').style.transform=`scaleX(${(index*secondsPerScene+elapsed/1000)/totalSeconds})`;
    // Deterministic native CSS microanimation sampling, independent of capture speed.
    document.getAnimations().forEach(animation=>{
      animation.pause();
      animation.currentTime=elapsed;
    });
  },{index,elapsed,secondsPerScene,totalSeconds});
}

function encode(frameDir, name, mobile) {
  const input=path.join(frameDir,'%04d.png');
  const gif=path.join(output,`${name}.gif`);
  // A per-export global palette retains small text and theme contrast; the
  // Bayer pattern and GIF delta rectangles keep the full loop compact.
  const filter='split[a][b];[a]palettegen=max_colors=160:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle';
  const result=spawnSync(ffmpeg,['-hide_banner','-loglevel','error','-y','-framerate',String(fps),'-i',input,'-filter_complex',filter,'-loop','0',gif],{stdio:'inherit'});
  if(result.status!==0) throw new Error('GIF encoding failed');
  if(!mobile) {
    const video=spawnSync(ffmpeg,['-hide_banner','-loglevel','error','-y','-framerate',String(fps),'-i',input,'-c:v','libx264','-crf','18','-preset','slow','-pix_fmt','yuv420p','-movflags','+faststart',path.join(output,`${name}.mp4`)],{stdio:'inherit'});
    if(video.status!==0) throw new Error('MP4 encoding failed');
  }
  console.log(JSON.stringify({file:gif,bytes:fs.statSync(gif).size,duration:totalSeconds}));
}

function recordArtifacts(name, dimensions, mobile) {
  const manifestPath=path.join(output,'animation-manifest.json');
  const manifest=fs.existsSync(manifestPath)?JSON.parse(fs.readFileSync(manifestPath,'utf8')):{
    source:'https://arxiv.org/pdf/2502.06355v3',
    walkthrough:'https://nousphera.github.io/projects/mpsl/',
    scene_order:['Tokenize','Send tokens','Predict','Local loss','Backward','Update'],
    seconds_per_scene:secondsPerScene,frames_per_second:fps,artifacts:[],
  };
  for(const extension of mobile?['gif']:['gif','mp4']) {
    const filename=`${name}.${extension}`;
    const bytes=fs.readFileSync(path.join(output,filename));
    const density=mobile?2:1;
    const entry={file:filename,bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex'),
      width:dimensions.width*density,height:dimensions.height*density,
      r_frame_rate:`${fps}/1`,nb_frames:String(fps*totalSeconds),duration_seconds:totalSeconds,
      ...(extension==='gif'?{loop:'infinite'}:{}),
    };
    manifest.artifacts=manifest.artifacts.filter(item=>item.file!==filename).concat(entry);
  }
  manifest.artifacts.sort((a,b)=>a.file.localeCompare(b.file));
  fs.writeFileSync(manifestPath,JSON.stringify(manifest,null,2)+'\n');
}

(async()=>{
  const browser=await chromium.launch({headless:true,...(chrome?{executablePath:chrome}:{})});
  try {
    for(const layout of layouts) for(const theme of themes) {
      const context=await browser.newContext({viewport:{width:layout==='desktop'?960:420,height:1000},deviceScaleFactor:layout==='desktop'?1:2,colorScheme:theme});
      const page=await context.newPage();
      const dimensions=await prepare(page,theme,layout);
      const name=`method-${theme}${layout==='mobile'?'-mobile':''}`;
      const frameDir=path.join(temp,name);
      fs.mkdirSync(frameDir);
      if(preview) {
        for(let index=0;index<6;index++) {
          await frame(page,index,4500);
          await page.screenshot({path:path.join(frameDir,`scene-${index}.png`)});
        }
        console.log(JSON.stringify({preview:frameDir,...dimensions}));
      } else {
        for(let n=0;n<fps*totalSeconds;n++) {
          const time=n/fps;
          const index=Math.floor(time/secondsPerScene);
          await frame(page,index,(time-index*secondsPerScene)*1000);
          await page.screenshot({path:path.join(frameDir,`${String(n).padStart(4,'0')}.png`)});
        }
        encode(frameDir,name,layout==='mobile');
        recordArtifacts(name,dimensions,layout==='mobile');
        console.log(JSON.stringify({layout,theme,...dimensions}));
      }
      await context.close();
    }
  } finally {
    await browser.close();
    if(preview || args.includes('--keep-frames')) console.log('Capture frames: '+temp);
    else fs.rmSync(temp,{recursive:true,force:true});
  }
})().catch(error=>{console.error(error);process.exitCode=1;});
