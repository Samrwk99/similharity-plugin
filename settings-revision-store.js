/**
 * Per-SillyTavern-user VectHare settings recovery snapshot.
 * Optimistic CAS, checksummed, atomic write + fsync. This is deliberately
 * separate from Qdrant and does not store API credentials or secret URLs.
 * Callers serialize inside each process AND through an atomic filesystem lock
 * on their common SillyTavern user-vector directory. Different filesystems
 * cannot coordinate: such deployments must not share writable Qdrant data.
 */
import { createHash } from 'node:crypto';
import path from 'node:path';
export const MAX_SETTINGS_BYTES=8*1024*1024;
const digest = data => createHash('sha256').update(JSON.stringify(data)).digest('hex');
const conflict=message=>Object.assign(new Error(message),{status:409});
export function createSettingsRevisionStore(fs){
    const locks=new Map();
    function filename(userVectors){return path.join(userVectors,'vecthare-settings-revisions','settings-v1.json');}
    async function serial(file,action){
        const old=locks.get(file)||Promise.resolve();
        let finish;const next=new Promise(r=>finish=r);locks.set(file,next);
        await old;
        try{return await action();}finally{finish();if(locks.get(file)===next)locks.delete(file);}
    }
    async function readFileOrEmpty(file){
        let contents;try{contents=await fs.readFile(file,'utf8');}
        catch(e){if(e.code==='ENOENT')return {revision:0,checksum:null,snapshot:null};throw e;}
        const row=JSON.parse(contents);
        if(row.schemaVersion!==1||!Number.isSafeInteger(row.revision)||row.revision<1||
            !row.snapshot||digest(row.snapshot)!==row.checksum)
            throw conflict('Settings checkpoint corrupt or checksum mismatch; manual recovery required');
        return row;
    }
    async function read(userVectors){return readFileOrEmpty(filename(userVectors));}
    async function readPrevious(userVectors){return readFileOrEmpty(`${filename(userVectors)}.previous`);}
    async function commit(userVectors,expectedRevision,snapshot){
        const file=filename(userVectors);
        return serial(file,async()=>{
            if(!Number.isSafeInteger(expectedRevision)||expectedRevision<0||
                !snapshot||Array.isArray(snapshot)||typeof snapshot!=='object')throw Object.assign(new Error('Invalid snapshot revision or data'),{status:400});
            const bytes=Buffer.byteLength(JSON.stringify(snapshot));
            if(bytes>MAX_SETTINGS_BYTES)throw Object.assign(new Error('Settings snapshot too large'),{status:413});
            // CAS must be indivisible ACROSS Node processes using the same ST
            // user directory, not just serialized inside this one process.
            await fs.mkdir(path.dirname(file),{recursive:true});
            const lock=`${file}.write-lock`;
            try{await fs.mkdir(lock);}catch(e){if(e.code==='EEXIST')throw conflict('Another settings commit is in progress; retry after it finishes');throw e;}
            try{
            const current=await read(userVectors);
            if(current.revision!==expectedRevision)throw conflict(`Stale settings revision: expected ${expectedRevision}, current ${current.revision}`);
            const row={schemaVersion:1,revision:expectedRevision+1,checksum:digest(snapshot),snapshot};
            await fs.mkdir(path.dirname(file),{recursive:true});
            // Preserve the last acknowledged recovery copy before replacing it.
            // Use copy+fsync rather than renaming the active file away: a crash
            // must never leave the current checkpoint missing.
            if(current.revision>0){
                const previous=`${file}.previous`;
                const oldTemp=`${previous}.${process.pid}.${Date.now()}.tmp`;
                try{
                    const oldHandle=await fs.open(oldTemp,'wx',0o600);
                    try{await oldHandle.writeFile(JSON.stringify(current));await oldHandle.sync();}
                    finally{await oldHandle.close();}
                    await fs.rename(oldTemp,previous);
                    const dir=await fs.open(path.dirname(file),'r');
                    try{await dir.sync();}finally{await dir.close();}
                }finally{await fs.rm(oldTemp,{force:true}).catch(()=>{});}
            }
            const temp=`${file}.${process.pid}.${Date.now()}.${Math.random().toString(16).slice(2)}.tmp`;
            try{
                const handle=await fs.open(temp,'wx',0o600);
                try{await handle.writeFile(JSON.stringify(row));await handle.sync();}finally{await handle.close();}
                await fs.rename(temp,file);
                // Acknowledgement is not sent until the directory entry is durable.
                const dir=await fs.open(path.dirname(file),'r');
                try{await dir.sync();}finally{await dir.close();}
            }finally{await fs.rm(temp,{force:true}).catch(()=>{});}
            const confirmed=await read(userVectors);
            if(confirmed.revision!==row.revision||confirmed.checksum!==row.checksum)
                throw conflict('Settings checkpoint read-back failed');
            return {revision:confirmed.revision,checksum:confirmed.checksum};
            }finally{await fs.rm(lock,{recursive:true,force:true});}
        });
    }
    return {read,readPrevious,commit};
}
