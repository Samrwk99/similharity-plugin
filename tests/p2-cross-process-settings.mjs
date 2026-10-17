import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {createSettingsRevisionStore} from '../settings-revision-store.js';
const dir=await fs.mkdtemp(path.join(os.tmpdir(),'vecthare-settings-cas-'));
try{
    const a=createSettingsRevisionStore(fs),b=createSettingsRevisionStore(fs);
    const simultaneous=await Promise.allSettled([
        a.commit(dir,0,{collections:{A:{model:'A'}},top_k:20}),
        b.commit(dir,0,{collections:{B:{model:'B'}},top_k:30}),
    ]);
    assert.equal(simultaneous.filter(v=>v.status==='fulfilled').length,1);
    assert.equal(simultaneous.filter(v=>v.status==='rejected').length,1);
    const live=await a.read(dir);
    assert.equal(live.revision,1);
    assert.ok(['A','B'].some(k=>live.snapshot.collections[k]));
    await assert.rejects(b.commit(dir,0,{collections:{C:{}}}),/Stale settings revision/);
    const next=await b.commit(dir,1,{...live.snapshot,settingWithZero:0,settingWithFalse:false});
    assert.equal(next.revision,2);
    const confirmed=await a.read(dir);
    assert.equal(confirmed.snapshot.settingWithZero,0);
    assert.equal(confirmed.snapshot.settingWithFalse,false);
    console.log('PASS distinct server processes cannot both commit the same settings revision');
    console.log('PASS stale snapshots do not overwrite newer collection settings');
    console.log('PASS verified zero/false settings survive revisioned persistence');
}finally{await fs.rm(dir,{recursive:true,force:true});}
