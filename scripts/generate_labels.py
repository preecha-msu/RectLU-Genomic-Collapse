cat << 'EOF' > generate_labels.py
import pysam
import numpy as np
import gzip
import time

def extract_truth_kmers_from_vcf(vcf_path, fasta_path, k_size=4, window_size=10):
    print(f"[{time.strftime('%H:%M:%S')}] สร้าง Truth Dictionary จาก {vcf_path}...")
    vcf = pysam.VariantFile(vcf_path)
    fasta = pysam.FastaFile(fasta_path)
    truth_kmers = set()
    
    for variant in vcf:
        chrom = variant.chrom
        pos = variant.pos
        start = max(0, pos - window_size)
        end = pos + window_size
        
        try:
            seq = fasta.fetch(chrom, start, end).upper()
            alt_seq = str(variant.alts[0]) if variant.alts else ""
            if len(alt_seq) > 1:
                seq = seq[:window_size] + alt_seq + seq[window_size:]
                
            for i in range(len(seq) - k_size + 1):
                truth_kmers.add(seq[i:i+k_size])
        except Exception:
            continue
            
    vcf.close()
    fasta.close()
    print(f"[{time.strftime('%H:%M:%S')}] -> สกัด Truth k-mers สำเร็จ: {len(truth_kmers)} รูปแบบ")
    return truth_kmers

def generate_labels_from_fastq(fastq_path, truth_kmers, k_size=4, max_reads=100000):
    print(f"[{time.strftime('%H:%M:%S')}] เริ่มแมป Labels ให้กับไฟล์ {fastq_path}...")
    labels = []
    
    with gzip.open(fastq_path, 'rt') as f:
        count = 0
        while True:
            _ = f.readline()          
            seq = f.readline().strip().upper() 
            _ = f.readline()          
            _ = f.readline()          
            
            if not seq:
                break
                
            is_variant = 0
            for i in range(len(seq) - k_size + 1):
                if seq[i:i+k_size] in truth_kmers:
                    is_variant = 1
                    break 
                    
            labels.append(is_variant)
            count += 1
            
            if max_reads and count >= max_reads:
                break
                
    labels_arr = np.array(labels, dtype=np.float32).reshape(-1, 1)
    num_variants = np.sum(labels_arr)
    num_total = len(labels_arr)
    print(f"[{time.strftime('%H:%M:%S')}] -> สำเร็จ! แมปทั้งหมด {num_total} reads (Variant: {num_variants}, ปกติ: {num_total - num_variants})")
    
    return labels_arr

if __name__ == "__main__":
    K_SIZE = 4
    # เพื่อป้องกัน Out of Memory ควรกำหนด MAX_READS ให้ตรงกับจำนวนข้อมูลที่ดึงไปตอนทำ X_train
    MAX_READS = 500000 
    
    BASE_DIR = "/mnt/e/TeachPree/Research_Academic_Services/GENOME"
    
    train_kmers = extract_truth_kmers_from_vcf(
        f"{BASE_DIR}/truth/train_truth.vcf.gz",
        f"{BASE_DIR}/reference/hg38.fa",
        k_size=K_SIZE
    )
    y_train = generate_labels_from_fastq(
        f"{BASE_DIR}/fastq_clean/train_R1.fastq.gz",
        train_kmers,
        k_size=K_SIZE,
        max_reads=MAX_READS
    )
    
    val_kmers = extract_truth_kmers_from_vcf(
        f"{BASE_DIR}/truth/val_truth.vcf.gz",
        f"{BASE_DIR}/reference/hg38.fa",
        k_size=K_SIZE
    )
    y_val = generate_labels_from_fastq(
        f"{BASE_DIR}/fastq_clean/val_R1.fastq.gz",
        val_kmers,
        k_size=K_SIZE,
        max_reads=MAX_READS // 5  # ชุด Val ดึงมาแค่ 20%
    )
    
    # บันทึกไฟล์ Labels ไว้ใช้ต่อในกระบวนการ Train
    np.savez_compressed(f"{BASE_DIR}/fastq_clean/y_labels_dataset.npz", y_train=y_train, y_val=y_val)
    print("บันทึกไฟล์ Label สำเร็จที่: fastq_clean/y_labels_dataset.npz")
EOF