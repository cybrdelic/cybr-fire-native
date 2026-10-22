from pathlib import Path
import tempfile,unittest,sys
sys.path.insert(0,str(Path(__file__).parent))
from checkpoint_guard import snapshot,verify,restore

class Tests(unittest.TestCase):
 def test_commit_missing_corruption_and_restore(self):
  with tempfile.TemporaryDirectory() as tmp:
   r=Path(tmp)/'run';s=Path(tmp)/'store';r.mkdir()
   (r/'state.npy').write_bytes(b'NUMERICAL_STATE_DATA')
   (r/'manifest.json').write_text('{"frames":1}')
   m=snapshot(r,s,['state.npy','manifest.json'],generation='gen001')
   self.assertEqual(verify(s)['generation'],'gen001')
   self.assertEqual(len(m['files']),2)
   try:snapshot(r,s,['missing.npy'],generation='gen002')
   except FileNotFoundError:pass
   else:self.fail('Missing checkpoint was incorrectly committed')
   self.assertEqual((s/'LATEST').read_text().strip(),'gen001')
   (r/'state.npy').unlink()
   out=Path(tmp)/'restore';restore(s,out)
   self.assertEqual((out/'state.npy').read_bytes(),b'NUMERICAL_STATE_DATA')
   src=s/'generations/gen001/payload/state.npy';src.write_bytes(b'CORRUPT')
   with self.assertRaises(ValueError):verify(s)
 def test_no_overwrite(self):
  with tempfile.TemporaryDirectory() as tmp:
   r=Path(tmp)/'run';s=Path(tmp)/'store';r.mkdir()
   (r/'a').write_text('a');snapshot(r,s,['a'],generation='g')
   with self.assertRaises(FileExistsError):restore(s,r)
if __name__=='__main__':unittest.main(verbosity=2)
