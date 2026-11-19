// GPU queries are read asynchronously only once the driver signals availability.
export class FrameTimings {
  constructor(gl) {
    this.gl=gl;
    this.ext=gl?.getExtension('EXT_disjoint_timer_query_webgl2');
    this.pending=[];
    this.clear();
  }
  clear() {
    this.values={solver:null,render:null};
    this.history={};
  }
  start() {
    if(!this.ext)return performance.now();
    const query=this.gl.createQuery();
    if(query)this.gl.beginQuery(this.ext.TIME_ELAPSED_EXT,query);
    return query;
  }
  finish(name,start,batch=null) {
    if(!this.ext){
      const ms=performance.now()-start;
      if(batch)batch.total+=ms;else this.record(name,ms);
    }else if(start){
      this.gl.endQuery(this.ext.TIME_ELAPSED_EXT);
      if(batch)batch.remaining++;
      this.pending.push({name,query:start,batch});
    }
  }
  measure(name,action) {
    if(this.pending.length>=6){action();return;}
    const start=this.start();
    try {action();} finally {this.finish(name,start);}
  }
  // Sample a single render stage per draw. Separate segments of the same
  // stage (emitter reduction and lighting cache, for example) are summed.
  measurePass(name,draw) {
    if(this.pending.length>=5){draw();return;}
    const batch={total:0,remaining:0};
    let start=null,seen=false;
    const marker=stage=>{
      if(start!==null){this.finish(name,start,batch);start=null;}
      if(stage===name){start=this.start();seen=true;}
    };
    try {draw(marker);} finally {
      marker(null);
      if(seen && !this.ext)this.record(name,batch.total);
    }
  }
  record(name,value) {
    if(!Number.isFinite(value))return;
    const previous=this.values[name];
    this.values[name]=previous==null?value:previous*.8+value*.2;
    const history=this.history[name]??(this.history[name]=[]);
    if(history.length===90)history.shift();
    history.push(value);
  }
  percentile95(name) {
    const samples=this.history[name];
    if(!samples || samples.length<8)return null;
    const sorted=samples.slice().sort((a,b)=>a-b);
    return sorted[Math.ceil(sorted.length*.95)-1];
  }
  poll() {
    if(!this.ext)return;
    const gl=this.gl;
    if(gl.getParameter(this.ext.GPU_DISJOINT_EXT)) {
      this.dispose();this.clear();return;
    }
    while(this.pending.length) {
      const {query,name,batch}=this.pending[0];
      if(!gl.getQueryParameter(query,gl.QUERY_RESULT_AVAILABLE))break;
      const ms=gl.getQueryParameter(query,gl.QUERY_RESULT)/1e6;
      if(batch){
        batch.total+=ms;
        if(--batch.remaining===0)this.record(name,batch.total);
      }else this.record(name,ms);
      gl.deleteQuery(query);this.pending.shift();
    }
  }
  dispose() {
    if(this.ext && !this.gl.isContextLost()) {
      for(const {query} of this.pending)this.gl.deleteQuery(query);
    }
    this.pending.length=0;
  }
  label() {
    return this.ext?'GPU elapsed time':this.gl?'CPU submission time':'CPU elapsed time';
  }
}
