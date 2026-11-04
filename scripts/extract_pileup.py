import pysam
import numpy as np

def base_to_onehot(base):
    """แปลงตัวอักษรเบสเป็น One-hot Array 4 แชนเนล (A, C, G, T)"""
    mapping = {
        'A': [1, 0, 0, 0], 'a': [1, 0, 0, 0],
        'C': [0, 1, 0, 0], 'c': [0, 1, 0, 0],
        'G': [0, 0, 1, 0], 'g': [0, 0, 1, 0],
        'T': [0, 0, 0, 1], 't': [0, 0, 0, 1]
    }
    return mapping.get(base, [0, 0, 0, 0]) # หากเป็น N หรืออื่นๆ ให้เป็น 0 ทั้งหมด

def generate_pileup_tensors(bam_path, vcf_path, fasta_path, window_size=101, max_depth=50, max_variants=10000):
    """สร้าง Pileup Matrix จากจุดกลายพันธุ์ใน VCF โดยใช้ pysam"""
    bam = pysam.AlignmentFile(bam_path, "rb")
    vcf = pysam.VariantFile(vcf_path)
    fasta = pysam.FastaFile(fasta_path)
    
    half_window = window_size // 2
    X_data = []
    y_labels = []
    
    print(f"กำลังประมวลผล Pileup จากไฟล์: {bam_path}")
    
    count = 0
    for variant in vcf:
        if count >= max_variants:
            break
            
        chrom = variant.chrom
        pos = variant.pos
        
        # กำหนดขอบเขตหน้าต่าง
        start_pos = pos - half_window
        end_pos = pos + half_window + 1
        
        # สร้างเมทริกซ์ว่างขนาด (Depth, Window_Size, Channels)
        pileup_matrix = np.zeros((max_depth, window_size, 4), dtype=np.float32)
        
        row_idx = 0
        # ดึง Reads ที่ซ้อนทับบริเวณหน้าต่างนี้
        for read in bam.fetch(chrom, start_pos, end_pos):
            if row_idx >= max_depth:
                break
                
            # ดึงลำดับเบสของ Read ที่ตำแหน่งนี้
            seq = read.query_alignment_sequence
            ref_positions = read.get_reference_positions(full_length=False)
            
            if not seq or not ref_positions:
                continue
                
            # แมปเบสลงในเมทริกซ์ตามตำแหน่งจริง
            for seq_idx, ref_pos in enumerate(ref_positions):
                if start_pos <= ref_pos < end_pos:
                    col_idx = ref_pos - start_pos
                    if seq_idx < len(seq):
                        base = seq[seq_idx]
                        pileup_matrix[row_idx, col_idx] = base_to_onehot(base)
            
            row_idx += 1
            
        X_data.append(pileup_matrix)
        y_labels.append(1) 
        
        count += 1
        if count % 1000 == 0:
            print(f"สกัด Pileup เสร็จแล้ว {count} ตำแหน่ง...")
            
    bam.close()
    vcf.close()
    fasta.close()
    
    return np.array(X_data, dtype=np.float32), np.array(y_labels, dtype=np.int32)

# ==========================================
# รัน Pipeline การสกัดข้อมูลและบันทึก (พิกัดสไตล์ Linux/WSL)
# ==========================================

# 1. ประมวลผลกลุ่ม Training Set
X_train_img, y_train_img = generate_pileup_tensors(
    bam_path="/mnt/e/TeachPree/Research_Academic_Services/GENOME/cram_data/train_align.bam",
    vcf_path="/mnt/e/TeachPree/Research_Academic_Services/GENOME/truth/train_truth.vcf.gz",
    fasta_path="/mnt/e/TeachPree/Research_Academic_Services/GENOME/reference/hg38.fa",
    max_variants=5000  # จำกัดจำนวนช่วงทดสอบระบบ
)

# 2. ประมวลผลกลุ่ม Validation Set
X_val_img, y_val_img = generate_pileup_tensors(
    bam_path="/mnt/e/TeachPree/Research_Academic_Services/GENOME/cram_data/val_align.bam",
    vcf_path="/mnt/e/TeachPree/Research_Academic_Services/GENOME/truth/val_truth.vcf.gz",
    fasta_path="/mnt/e/TeachPree/Research_Academic_Services/GENOME/reference/hg38.fa",
    max_variants=1000
)

# บันทึกข้อมูลเข้า .npz โดยตรงกลับไปยังไดรฟ์ E ของ Windows
np.savez_compressed(
    '/mnt/e/TeachPree/Research_Academic_Services/GENOME/cram_data/pileup_features_dataset.npz', 
    X_train=X_train_img, y_train=y_train_img,
    X_val=X_val_img, y_val=y_val_img
)

print("\nกระบวนการสกัด Pileup Image เสร็จสมบูรณ์! ไฟล์ถูกบันทึกไว้ที่ cram_data/pileup_features_dataset.npz")
print(f"มิติของเทนเซอร์ Training ชุดนี้คือ: {X_train_img.shape}")