/**
 * Crash-detectable incremental manifests for the Similharity sidecar.
 * The .json file is a complete checkpoint; .journal is newline-delimited,
 * contiguous compare-and-swap revisions. A partial final line is NOT ignored.
 * Read, set and patch must all run under the caller's per-file write lock.
 */
export function createProfileJournal(fs, writeSnapshot, options={}) {
    const cache=new Map();
    const compactEvery=options.compactEvery??256;
    const maxJournalBytes=options.maxJournalBytes??4*1024*1024;
    const maxPatchBytes=options.maxPatchBytes??15*1024*1024;
    const conflict=message=>Object.assign(new Error(message),{status:409});
    const apply=(state,delta)=>{
        if (!state?.manifest || state.manifest.schemaVersion!==1 ||
            delta.revision !== (state.manifest.revision??0)+1 ||
            delta.expectedUpdatedAt !== (state.manifest.updatedAt??null) ||
            delta.expectedRevision !== (state.manifest.revision??0) ||
            delta.chatUUID !== state.manifest.chatUUID)
            throw conflict('Manifest journal revision or chat identity does not match its checkpoint');
        const entries=new Map(state.manifest.entries.map(row=>[String(row.hash),row]));
        for(const hash of delta.removedHashes)entries.delete(String(hash));
        for(const row of delta.upserts)entries.set(String(row.hash),row);
        return {...state,manifest:{...state.manifest,entries:[...entries.values()],
            revision:delta.revision,updatedAt:delta.updatedAt}};
    };
    const sig=stat=>`${stat.ino}:${stat.size}:${stat.mtimeMs}`;
    async function diskSignature(target){
        const base=sig(await fs.stat(target));
        let journalBytes=0;
        try{journalBytes=(await fs.stat(`${target}.journal`)).size;}
        catch(error){if(error.code!=='ENOENT')throw error;}
        return {base,journalBytes};
    }
    async function load(target){
        if(cache.has(target)){
            const current=await diskSignature(target);
            const previous=cache.get(target);
            if(current.base!==previous.snapshotSignature || current.journalBytes!==previous.bytes)
                throw conflict('Profile sidecar changed outside this process; reconcile before another write');
            return previous;
        }
        let state=JSON.parse(await fs.readFile(target,'utf8'));
        let journal='';
        try{journal=await fs.readFile(`${target}.journal`,'utf8');}
        catch(error){if(error.code!=='ENOENT')throw error;}
        if(journal && !journal.endsWith('\n'))
            throw conflict('Manifest journal ends in a partial write; explicit recovery required');
        let patches=0;
        for(const line of journal.split('\n')){
            if(!line)continue;
            let delta;
            try{delta=JSON.parse(line);}catch{throw conflict('Malformed manifest journal; recovery required');}
            if(!Number.isSafeInteger(delta.revision))throw conflict('Invalid journal revision');
            // Compaction may have completed before its old journal was removed.
            if(delta.revision <= (state.manifest?.revision??0))continue;
            state=apply(state,delta);
            patches++;
        }
        const found={state,patches,bytes:Buffer.byteLength(journal),
            snapshotSignature:sig(await fs.stat(target))};
        cache.set(target,found);
        return found;
    }
    async function set(target,newState){
        const old=await load(target).catch(error=>{
            if(error.code==='ENOENT')return null;
            throw error;
        });
        if(old?.state?.profile?.recipe && JSON.stringify(old.state.profile.recipe)!==JSON.stringify(newState.profile?.recipe))
            throw conflict('Existing collection embedding recipe is immutable; create a staged replacement instead');
        if(old?.state?.manifest && newState.manifest &&
            (newState.manifest.revision??0)===(old.state.manifest.revision??0) &&
            ((newState.manifest.updatedAt??null)!==(old.state.manifest.updatedAt??null) ||
             JSON.stringify(newState.manifest.entries)!==JSON.stringify(old.state.manifest.entries)))
            throw conflict('Same manifest revision has different contents; reconcile instead of overwriting');
        if(old?.state?.manifest && newState.manifest &&
            (newState.manifest.revision??0) < (old.state.manifest.revision??0))
            throw conflict('Refusing to overwrite a newer committed manifest with an older snapshot');
        if(old?.state?.manifest && !newState.manifest)
            throw conflict('Refusing to erase an existing manifest with a profile-only save');
        // An old journal may remain if a process crashes after checkpoint rename.
        // The replay algorithm ignores revisions <= the new checkpoint revision.
        await writeSnapshot(target,JSON.stringify(newState));
        await fs.rm(`${target}.journal`,{force:true});
        cache.set(target,{state:newState,patches:0,bytes:0,
            snapshotSignature:sig(await fs.stat(target))});
        return newState;
    }
    async function patch(target,delta){
        const record=await load(target);
        if(record.compactionOverLimit)
            throw Object.assign(new Error('Manifest snapshot is at its safe size limit; archive or rebuild into separate collections'),{status:413});
        if(record.state.key!==delta.key || record.state.profile?.schemaVersion!==1 ||
            record.state.manifest?.schemaVersion!==1)
            throw conflict('Saved profile/manifest missing or inconsistent; repair first');
        const next=apply(record.state,delta);
        const line=JSON.stringify(delta)+'\n';
        const size=Buffer.byteLength(line);
        if(size>maxPatchBytes)throw Object.assign(new Error('Manifest patch exceeds size limit'),{status:413});
        // Durably append before acknowledging vector checkpoints. The lock held
        // by the caller prevents another update from seeing an uncommitted rev.
        const fd=await fs.open(`${target}.journal`,'a');
        try {await fd.writeFile(line);await fd.sync();}
        finally {await fd.close();}
        record.state=next;
        record.patches++;
        record.bytes+=size;
        if(record.patches>=compactEvery || record.bytes>=maxJournalBytes){
            // Compaction failures do not invalidate an already-fsynced patch.
            try {
                await writeSnapshot(target,JSON.stringify(next));
                await fs.rm(`${target}.journal`,{force:true});
                record.patches=0;record.bytes=0;
                record.snapshotSignature=sig(await fs.stat(target));
            } catch(error){
                if(error.status===413)record.compactionOverLimit=true;
                console.warn('[Similharity] Manifest compaction postponed:',error.message);
            }
        }
        return {success:true,revision:next.manifest.revision,
            updatedAt:next.manifest.updatedAt,entries:next.manifest.entries.length};
    }
    return {load,set,patch};
}
