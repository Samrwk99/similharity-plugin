/**
 * Cross-tab, cross-process job coordination for a single SillyTavern user.
 * The filesystem containing req.user.directories.vectors is the authority.
 * All processes MUST mount that same filesystem for these guarantees to hold.
 * No expiry-based lock stealing: a dead/stalled writer must be reconciled
 * explicitly before another worker can mutate the same database.
 * Status contains no chat content, vectors or credentials.
 */
import path from 'node:path';
import { randomUUID } from 'node:crypto';

const bad=(message,status=409)=>Object.assign(new Error(message),{status});
const safeName=(s)=>String(s||'').replace(/[^a-zA-Z0-9 _.-]/g,'').slice(0,90);
export function createSharedOperationStore(fs,{clock=Date.now}={}) {
    const userRoot=vectors=>path.join(vectors,'vecthare-operation-coordinator-v1');
    const activeDir=root=>path.join(root,'active');
    const queueDir=root=>path.join(root,'queued');
    const archiveDir=root=>path.join(root,'history');
    const read=async file=>{try{return JSON.parse(await fs.readFile(file,'utf8'));}catch(e){if(e.code==='ENOENT')return null;throw e;}};
    async function atomic(file,obj) {
        const tmp=`${file}.${randomUUID()}.tmp`;
        const fd=await fs.open(tmp,'wx',0o600);
        try{await fd.writeFile(JSON.stringify(obj));await fd.sync();}finally{await fd.close();}
        try{await fs.rename(tmp,file);}finally{await fs.rm(tmp,{force:true}).catch(()=>{});}
    }
    async function prepare(root){await fs.mkdir(queueDir(root),{recursive:true});await fs.mkdir(archiveDir(root),{recursive:true});}
    // Every queue transition is protected by one atomic filesystem mutex.
    // Concurrent processes must not both inspect an empty queue and start
    // different jobs (or accept two same-resource jobs) at the same time.
    // A crash leaves the mutex in place: never steal it on elapsed time.
    async function withTransition(root, action) {
        const lock=path.join(root,'transition-lock');
        let acquired=false;
        for(let attempt=0;attempt<100;attempt++) {
            try { await fs.mkdir(lock);acquired=true;break; }
            catch(error) {
                if(error.code!=='EEXIST')throw error;
                await new Promise(resolve=>setTimeout(resolve,10));
            }
        }
        if(!acquired)throw bad('Operation coordinator transition is locked; an interrupted transition may require administrator reconciliation',423);
        try { return await action(); }
        finally { await fs.rm(lock,{recursive:true,force:true}); }
    }
    async function queued(root){
        let names=[];try{names=await fs.readdir(queueDir(root));}catch(e){if(e.code!=='ENOENT')throw e;}
        const jobs=[];
        for(const name of names.filter(n=>n.endsWith('.json')).sort()){
            const job=await read(path.join(queueDir(root),name));if(job)jobs.push({...job,_filename:name});
        }
        return jobs;
    }
    const publicJob=j=>j&&({id:j.id,label:j.label,category:j.category,resourceKey:j.resourceKey,
        phase:j.phase,detail:j.detail,completed:j.completed||0,total:j.total||0,unit:j.unit||'items',
        startedAt:j.startedAt,updatedAt:j.updatedAt,requestedAt:j.requestedAt,
        cancelRequested:!!j.cancelRequested,recoveryPrepared:!!j.recoveryPrepared,source:'shared'});
    async function current(root){return read(path.join(activeDir(root),'job.json'));}
    async function start(vectors,data={}){
        if(!vectors)throw bad('Authenticated user storage is required',401);
        const root=userRoot(vectors);await prepare(root);
        return withTransition(root,async()=>{
        const now=clock();const id=randomUUID(), token=randomUUID();
        const job={id,token,label:safeName(data.label||'VectHare operation'),category:safeName(data.category||'write'),
            resourceKey:safeName(data.resourceKey||''),detail:'',unit:'items',completed:0,total:0,
            requestedAt:now,updatedAt:now,startedAt:null,phase:'queued'};
        // We refuse all new writes if the previous owner disappeared. A prior
        // orphaned active directory is *not* safe to steal just due to time.
        const alreadyQueued=await queued(root);
        const running=await current(root);
        // Bounded queue and same-target coalescing prevent repeated button
        // presses from overwhelming embedding servers or filling user storage.
        const sameTarget=j=>j && job.resourceKey && j.resourceKey===job.resourceKey && j.category===job.category;
        if([running,...alreadyQueued].some(sameTarget))
            throw bad('This chat/collection job is already running or queued. View it in Live Status.');
        if(alreadyQueued.length>=16)throw bad('VectHare backend job queue is full (16 pending); cancel an old job first',429);
        if(data.queue===false && (alreadyQueued.length || running))throw bad('Another backend job is running; automatic work was not queued');
        if(alreadyQueued.length || running) {
            const filename=`${String(now).padStart(15,'0')}-${id}.json`;
            await atomic(path.join(queueDir(root),filename),job);
            return {id,token,phase:'queued'};
        }
        try{
            await fs.mkdir(activeDir(root));
            job.phase='running';job.startedAt=now;
            await atomic(path.join(activeDir(root),'job.json'),job);
            return {id,token,phase:'running'};
        }catch(e){
            if(e.code!=='EEXIST')throw e;
            const filename=`${String(now).padStart(15,'0')}-${id}.json`;
            await atomic(path.join(queueDir(root),filename),job);
            return {id,token,phase:'queued'};
        }
        });
    }
    async function authorize(vectors,id,token){
        const root=userRoot(vectors),job=await current(root);
        if(!job||job.id!==id||job.token!==token)throw bad('Operation not active or lease token invalid');
        return {root,job};
    }
    async function status(vectors,id,token){
        if(!vectors)throw bad('Authenticated user storage is required',401);
        const root=userRoot(vectors), active=await current(root);
        if(active?.id===id){if(active.token!==token)throw bad('Invalid operation token',403);
            const cancel=await read(path.join(activeDir(root),'cancel.json'));
            return {...publicJob(active),cancelRequested:!!cancel};}
        const match=(await queued(root)).find(j=>j.id===id);
        if(match){if(match.token!==token)throw bad('Invalid operation token',403);return publicJob(match);}
        const ended=await read(path.join(archiveDir(root),`${id}.json`));
        if(ended){if(ended.token!==token)throw bad('Invalid operation token',403);return publicJob(ended);}
        throw bad('Operation is not known',404);
    }
    async function list(vectors){
        if(!vectors)throw bad('Authenticated user storage is required',401);
        const root=userRoot(vectors), active=await current(root);
        const jobs=[];
        if(active){const cancel=await read(path.join(activeDir(root),'cancel.json'));
            const job=publicJob(active);job.cancelRequested=!!cancel;
            const closing=await read(path.join(activeDir(root),'closing'));
            if(closing?.reason==='explicit-recovery'){job.phase='interrupted';job.recoveryPrepared=true;}
            else if(job.phase==='running' && clock()-job.updatedAt>45000)job.phase='interrupted';
            jobs.push(job);
        }
        jobs.push(...(await queued(root)).map(publicJob));
        return {operations:jobs};
    }
    async function heartbeat(vectors,id,token,patch={}){
        const {root,job}=await authorize(vectors,id,token);
        const cancel=await read(path.join(activeDir(root),'cancel.json'));
        if(await read(path.join(activeDir(root),'closing')))throw bad('Operation finalisation is blocked; manual reconciliation required');
        if(job.phase!=='running' && job.phase!=='paused')throw bad('Cannot renew a non-running operation');
        job.updatedAt=clock();
        if(Number.isFinite(patch.completed))job.completed=Math.max(0,patch.completed);
        if(Number.isFinite(patch.total))job.total=Math.max(0,patch.total);
        if(typeof patch.detail==='string')job.detail=safeName(patch.detail);
        if(typeof patch.unit==='string')job.unit=safeName(patch.unit);
        // The lease is valid only while this owner is current. The consumer
        // rechecks cancellation at safe checkpoints, never during a vector write.
        await atomic(path.join(activeDir(root),'job.json'),job);
        return {...publicJob(job),cancelRequested:!!cancel};
    }
    async function cancel(vectors,id){
        if(!vectors)throw bad('Authenticated user storage is required',401);
        const root=userRoot(vectors);
        return withTransition(root,async()=>{
        const job=await current(root);
        if(job?.id===id){
            await atomic(path.join(activeDir(root),'cancel.json'),{requestedAt:clock()});
            return {requested:true,phase:'stopping'};
        }
        for(const q of await queued(root))if(q.id===id){
            const file=path.join(queueDir(root),q._filename);
            const archive=path.join(archiveDir(root),`${id}.json`);
            // Rename is the atomic claim: only cancellation OR promotion wins.
            try{await fs.rename(file,archive);}
            catch(e){if(e.code==='ENOENT')throw bad('Operation changed during cancellation; refresh Live Status and retry');throw e;}
            const ended={...q,phase:'stopped',detail:'Cancelled while queued',updatedAt:clock()};
            delete ended._filename;
            await atomic(archive,ended);
            return {requested:true,phase:'stopped'};
        }
        throw bad('Operation not found or already finished',404);
        });
    }
    async function complete(vectors,id,token,phase='completed',detail=''){
        const {root}=await authorize(vectors,id,token);
        return withTransition(root,async()=>{
        if(!['completed','failed','stopped'].includes(phase))throw bad('Invalid completion phase',400);
        // A second completion may never remove a newly promoted job. The
        // completion lock is an atomic claim shared by server processes.
        const finishing=path.join(root,'completing');
        try{await fs.mkdir(finishing);}catch(e){if(e.code==='EEXIST')throw bad('Another completion is already in progress');throw e;}
        try{
            const live=await current(root);
            if(!live||live.id!==id||live.token!==token)throw bad('Operation ownership changed before completion');
            await atomic(path.join(activeDir(root),'closing'),{requestedAt:clock()});
            // The HTTP client may have disconnected while a backend was still
            // writing. Never hand the lock to a new writer until *every* server
            // mutation explicitly finished. A crashed process leaves its marker
            // behind for manual recovery rather than risking concurrent writes.
            const inflight=path.join(activeDir(root),'inflight');
            let files=[];
            try{files=await fs.readdir(inflight);}catch(e){if(e.code!=='ENOENT')throw e;}
            if(files.length)throw bad('Backend requests are still in flight; lease retained for safe recovery');
            const ended={...live,phase,detail:safeName(detail),updatedAt:clock()};
            await atomic(path.join(archiveDir(root),`${id}.json`),ended);
            await fs.rm(activeDir(root),{recursive:true,force:true});
            // Promote in deterministic queue order, never overwriting active.
            for(const next of await queued(root)){
                try{await fs.mkdir(activeDir(root));}
                catch(e){if(e.code==='EEXIST')break;throw e;}
                try{
                    // The queued-file rename races atomically with cancellation.
                    await fs.rename(path.join(queueDir(root),next._filename),path.join(activeDir(root),'job.json'));
                    const promoted={...next,phase:'running',startedAt:clock(),updatedAt:clock()};
                    delete promoted._filename;
                    await atomic(path.join(activeDir(root),'job.json'),promoted);
                    break;
                }catch(e){
                    await fs.rm(activeDir(root),{recursive:true,force:true});
                    if(e.code!=='ENOENT')throw e;
                }
            }
            return {finished:true};
        }finally{
            // An unexpected crash leaves `completing` intentionally blocking
            // successors. Never silently discard evidence of a partial handoff.
            await fs.rm(finishing,{recursive:true,force:true});
        }
        });
    }
    /** Explicit recovery of a stalled owner on ONE shared-storage SillyTavern deployment.
     * Revokes the old ownership BEFORE inspection. Never clears an in-flight
     * request marker, nor blindly steals a job merely because its tab vanished.
     * This only resumes the queue; clients must reconcile actual DB state.
     */
    async function recover(vectors,id,confirmation){
        if(!vectors)throw bad('Authenticated user storage is required',401);
        if(!['FENCE INTERRUPTED JOB','RECONCILED AND RELEASE'].includes(confirmation))
            throw bad('Recovery requires an explicit two-stage confirmation',400);
        const root=userRoot(vectors);await prepare(root);
        return withTransition(root,async()=>{
            const job=await current(root);
            if(!job||job.id!==id)throw bad('Interrupted job not found',404);
            const oldClosing=await read(path.join(activeDir(root),'closing'));
            if(clock()-job.updatedAt<120000 && oldClosing?.reason!=='explicit-recovery')
                throw bad('Writer was recently active. Cancel it and allow it to finish safely before recovery',423);
            const inflight=path.join(activeDir(root),'inflight');
            let entries=[];
            try{entries=await fs.readdir(inflight);}catch(e){if(e.code!=='ENOENT')throw e;}
            if(entries.length && confirmation==='RECONCILED AND RELEASE')
                throw bad('Backend writes are or were in flight. Cannot safely release the queue before backend reconciliation',423);
            // Freeze the old token before any asynchronous work or queue handoff.
            // Every plugin write and heartbeat rechecks this barrier.
            if(oldClosing && oldClosing.reason!=='explicit-recovery')
                throw bad('Operation is already finalising; use manual administrator inspection',423);
            if(!oldClosing)await atomic(path.join(activeDir(root),'closing'),{requestedAt:clock(),reason:'explicit-recovery'});
            // Heartbeat and mutation arrival can race with closing. Allow a
            // short drain; neither route may start a new backend write now.
            await new Promise(resolve=>setTimeout(resolve,50));
            try{entries=await fs.readdir(inflight);}catch(e){if(e.code!=='ENOENT')throw e;}
            if(entries.length && confirmation==='RECONCILED AND RELEASE')
                throw bad('In-flight write remains during recovery; lock retained for inspection',423);
            const live=await current(root);
            if(!live||live.id!==id||live.token!==job.token)
                throw bad('Operation owner changed during recovery',423);
            if(confirmation==='FENCE INTERRUPTED JOB')
                return {fenced:true,recovered:false,requiresReconciliation:true,
                    message:'Old writer fenced. Inspect/repair affected collections before releasing queued work.'};
            if(!oldClosing || oldClosing.reason!=='explicit-recovery')
                throw bad('First fence the interrupted writer, then reconcile database state before release',423);
            // Force future operations to reconcile DB and profile journal:
            // recovered = interrupted, never misreported as completed.
            await atomic(path.join(archiveDir(root),`${id}.json`),
                {...live,phase:'interrupted',updatedAt:clock(),detail:'Recovery acknowledged; database reconciliation required before resuming'});
            await fs.rm(activeDir(root),{recursive:true,force:true});
            for(const next of await queued(root)){
                try{await fs.mkdir(activeDir(root));}
                catch(e){if(e.code==='EEXIST')break;throw e;}
                try{
                    await fs.rename(path.join(queueDir(root),next._filename),path.join(activeDir(root),'job.json'));
                    const promoted={...next,phase:'running',startedAt:clock(),updatedAt:clock()};
                    delete promoted._filename;
                    await atomic(path.join(activeDir(root),'job.json'),promoted);
                    break;
                }catch(e){await fs.rm(activeDir(root),{recursive:true,force:true});if(e.code!=='ENOENT')throw e;}
            }
            return {recovered:true,previousPhase:'interrupted',requiresReconciliation:true};
        });
    }
    // Native /api/vector requests do not traverse plugin middleware. The
    // browser must reserve a durable server-side in-flight marker *before*
    // dispatching them; a lost tab leaves a blocker for manual reconciliation.
    async function beginNative(vectors,id,token){
        const valid=await verifyMutation(vectors,id,token);
        if(!valid.active)throw bad('A verified active operation is required for native writes',423);
        const root=userRoot(vectors),dir=activeDir(root);
        const inflight=path.join(dir,'inflight');await fs.mkdir(inflight,{recursive:true});
        const markerId=randomUUID();
        await atomic(path.join(inflight,`${markerId}.json`),{id,kind:'native',startedAt:clock()});
        const live=await current(root);
        if(await read(path.join(dir,'closing')) || live?.id!==id || live?.token!==token){
            await fs.rm(path.join(inflight,`${markerId}.json`),{force:true});
            throw bad('Operation ownership changed before native request; native write refused',423);
        }
        return {markerId};
    }
    async function endNative(vectors,id,token,markerId){
        const {root}=await authorize(vectors,id,token);
        if(!/^[0-9a-f-]{36}$/i.test(String(markerId||'')))throw bad('Invalid native marker',400);
        const marker=path.join(activeDir(root),'inflight',`${markerId}.json`);
        const data=await read(marker);
        if(!data||data.kind!=='native'||data.id!==id)throw bad('Native mutation marker not found',404);
        await fs.rm(marker,{force:true});
        return {released:true};
    }
    async function beginMutation(vectors,id,token){
        let implicit=null;
        const validation=await verifyMutation(vectors,id,token);
        if(!validation.active){
            // Even one-shot edits and administrative writes must use the SAME
            // durable per-user coordinator as long-running operations. The
            // old unrestricted no-active-job fallthrough allowed two server
            // processes to mutate the backend concurrently.
            if(id||token)throw bad('Stale operation credentials cannot authorise a new write',423);
            implicit=await start(vectors,{label:'VectHare one-shot backend write',category:'one-shot',
                resourceKey:'single backend mutation',queue:false});
            id=implicit.id;token=implicit.token;
        }
        const root=userRoot(vectors),dir=activeDir(root),closed=path.join(dir,'closing');
        if(await read(closed))throw bad('Operation is finalising; new backend writes are prohibited',423);
        const inflight=path.join(dir,'inflight');await fs.mkdir(inflight,{recursive:true});
        const marker=path.join(inflight,`${randomUUID()}.json`);
        const fd=await fs.open(marker,'wx',0o600);
        try{await fd.writeFile(JSON.stringify({id,startedAt:clock()}));await fd.sync();}
        finally{await fd.close();}
        // Close the finalise-vs-request race before allowing a new mutation.
        const stillOwner=await current(root);
        if(await read(closed) || stillOwner?.id!==id || stillOwner?.token!==token){
            await fs.rm(marker,{force:true});
            throw bad('Operation ownership changed while request entered; write refused',423);
        }
        let released=false;
        return async()=>{
            if(released)return;
            released=true;
            await fs.rm(marker,{force:true});
            if(implicit)await complete(vectors,implicit.id,implicit.token,'completed','One-shot request settled');
        };
    }
    async function verifyMutation(vectors,id,token){
        if(!vectors)throw bad('Authenticated user storage is required',401);
        const active=await current(userRoot(vectors));
        if(!active)return {active:false};
        if(await read(path.join(activeDir(userRoot(vectors)),'closing')))
            throw bad('Operation is fenced for finalisation or recovery; writes blocked',423);
        if(!id||!token||active.id!==id||active.token!==token)
            throw bad('A different VectHare job owns the backend. Use Live Status to view or cancel it.',423);
        if(await read(path.join(activeDir(userRoot(vectors)),'cancel.json')))
            throw bad('Shared operation cancellation is pending; refusing a new vector write',423);
        return {active:true,id:active.id};
    }
    return {start,status,list,heartbeat,cancel,complete,recover,verifyMutation,beginMutation,beginNative,endNative};
}
