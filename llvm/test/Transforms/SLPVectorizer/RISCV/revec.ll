; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -mtriple=riscv64 -mcpu=sifive-x280 -passes=slp-vectorizer -S -slp-revec -slp-max-reg-size=1024 -slp-threshold=-100 %s | FileCheck %s

define i32 @test() {
; CHECK-LABEL: @test(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[IF_END_I87:%.*]]
; CHECK:       if.end.i87:
; CHECK-NEXT:    [[TMP0:%.*]] = call <4 x i32> @llvm.masked.gather.v4i32.v4p0(<4 x ptr> getelementptr (i32, <4 x ptr> <ptr inttoptr (i64 64036 to ptr), ptr inttoptr (i64 64036 to ptr), ptr inttoptr (i64 64064 to ptr), ptr inttoptr (i64 64064 to ptr)>, <4 x i64> <i64 0, i64 1, i64 0, i64 1>), i32 4, <4 x i1> splat (i1 true), <4 x i32> poison)
; CHECK-NEXT:    [[TMP2:%.*]] = call <4 x i32> @llvm.vector.insert.v4i32.v2i32(<4 x i32> poison, <2 x i32> zeroinitializer, i64 2)
; CHECK-NEXT:    [[TMP3:%.*]] = shufflevector <4 x i32> [[TMP0]], <4 x i32> [[TMP2]], <4 x i32> <i32 0, i32 1, i32 6, i32 7>
; CHECK-NEXT:    switch i32 0, label [[SW_BB509_I:%.*]] [
; CHECK-NEXT:      i32 1, label [[SW_BB509_I]]
; CHECK-NEXT:      i32 0, label [[IF_THEN458_I:%.*]]
; CHECK-NEXT:    ]
; CHECK:       if.then458.i:
; CHECK-NEXT:    br label [[SW_BB509_I]]
; CHECK:       sw.bb509.i:
; CHECK-NEXT:    [[TMP4:%.*]] = phi <4 x i32> [ [[TMP0]], [[IF_THEN458_I]] ], [ [[TMP3]], [[IF_END_I87]] ], [ [[TMP3]], [[IF_END_I87]] ]
; CHECK-NEXT:    ret i32 0
;
entry:
  %getelementptr0 = getelementptr i8, ptr null, i64 64036
  %getelementptr1 = getelementptr i8, ptr null, i64 64064
  br label %if.end.i87

if.end.i87:                                       ; preds = %entry
  %0 = load <2 x i32>, ptr %getelementptr0, align 4
  %1 = load <2 x i32>, ptr %getelementptr1, align 8
  switch i32 0, label %sw.bb509.i [
  i32 1, label %sw.bb509.i
  i32 0, label %if.then458.i
  ]

if.then458.i:                                     ; preds = %if.end.i87
  br label %sw.bb509.i

sw.bb509.i:                                       ; preds = %if.then458.i, %if.end.i87, %if.end.i87
  %4 = phi <2 x i32> [ %0, %if.then458.i ], [ %0, %if.end.i87 ], [ %0, %if.end.i87 ]
  %5 = phi <2 x i32> [ %1, %if.then458.i ], [ zeroinitializer, %if.end.i87 ], [ zeroinitializer, %if.end.i87 ]
  ret i32 0
}

define void @test2() {
; CHECK-LABEL: @test2(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr i8, ptr null, i64 132
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr i8, ptr null, i64 200
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr i8, ptr null, i64 300
; CHECK-NEXT:    [[TMP3:%.*]] = load <8 x float>, ptr [[TMP1]], align 4
; CHECK-NEXT:    [[TMP4:%.*]] = load <8 x float>, ptr [[TMP2]], align 4
; CHECK-NEXT:    [[TMP5:%.*]] = load <16 x float>, ptr [[TMP0]], align 4
; CHECK-NEXT:    [[TMP6:%.*]] = call <32 x float> @llvm.vector.insert.v32f32.v8f32(<32 x float> poison, <8 x float> [[TMP4]], i64 0)
; CHECK-NEXT:    [[TMP7:%.*]] = call <32 x float> @llvm.vector.insert.v32f32.v8f32(<32 x float> [[TMP6]], <8 x float> [[TMP3]], i64 8)
; CHECK-NEXT:    [[TMP8:%.*]] = call <32 x float> @llvm.vector.insert.v32f32.v16f32(<32 x float> [[TMP7]], <16 x float> [[TMP5]], i64 16)
; CHECK-NEXT:    [[TMP9:%.*]] = fpext <32 x float> [[TMP8]] to <32 x double>
; CHECK-NEXT:    [[TMP10:%.*]] = call <32 x double> @llvm.vector.insert.v32f64.v8f64(<32 x double> poison, <8 x double> zeroinitializer, i64 0)
; CHECK-NEXT:    [[TMP11:%.*]] = call <32 x double> @llvm.vector.insert.v32f64.v8f64(<32 x double> [[TMP10]], <8 x double> zeroinitializer, i64 8)
; CHECK-NEXT:    [[TMP12:%.*]] = call <32 x double> @llvm.vector.insert.v32f64.v8f64(<32 x double> [[TMP11]], <8 x double> zeroinitializer, i64 16)
; CHECK-NEXT:    [[TMP13:%.*]] = call <32 x double> @llvm.vector.insert.v32f64.v8f64(<32 x double> [[TMP12]], <8 x double> zeroinitializer, i64 24)
; CHECK-NEXT:    [[TMP14:%.*]] = fadd <32 x double> [[TMP13]], [[TMP9]]
; CHECK-NEXT:    [[TMP15:%.*]] = fptrunc <32 x double> [[TMP14]] to <32 x float>
; CHECK-NEXT:    [[TMP16:%.*]] = call <32 x float> @llvm.vector.insert.v32f32.v8f32(<32 x float> poison, <8 x float> zeroinitializer, i64 0)
; CHECK-NEXT:    [[TMP17:%.*]] = call <32 x float> @llvm.vector.insert.v32f32.v8f32(<32 x float> [[TMP16]], <8 x float> zeroinitializer, i64 8)
; CHECK-NEXT:    [[TMP18:%.*]] = call <32 x float> @llvm.vector.insert.v32f32.v8f32(<32 x float> [[TMP17]], <8 x float> zeroinitializer, i64 16)
; CHECK-NEXT:    [[TMP19:%.*]] = call <32 x float> @llvm.vector.insert.v32f32.v8f32(<32 x float> [[TMP18]], <8 x float> zeroinitializer, i64 24)
; CHECK-NEXT:    [[TMP20:%.*]] = fcmp ogt <32 x float> [[TMP19]], [[TMP15]]
; CHECK-NEXT:    ret void
;
entry:
  %0 = getelementptr i8, ptr null, i64 132
  %1 = getelementptr i8, ptr null, i64 164
  %2 = getelementptr i8, ptr null, i64 200
  %3 = getelementptr i8, ptr null, i64 300
  %4 = load <8 x float>, ptr %0, align 4
  %5 = load <8 x float>, ptr %1, align 4
  %6 = load <8 x float>, ptr %2, align 4
  %7 = load <8 x float>, ptr %3, align 4
  %8 = fpext <8 x float> %4 to <8 x double>
  %9 = fpext <8 x float> %5 to <8 x double>
  %10 = fpext <8 x float> %6 to <8 x double>
  %11 = fpext <8 x float> %7 to <8 x double>
  %12 = fadd <8 x double> zeroinitializer, %8
  %13 = fadd <8 x double> zeroinitializer, %9
  %14 = fadd <8 x double> zeroinitializer, %10
  %15 = fadd <8 x double> zeroinitializer, %11
  %16 = fptrunc <8 x double> %12 to <8 x float>
  %17 = fptrunc <8 x double> %13 to <8 x float>
  %18 = fptrunc <8 x double> %14 to <8 x float>
  %19 = fptrunc <8 x double> %15 to <8 x float>
  %20 = fcmp ogt <8 x float> zeroinitializer, %16
  %21 = fcmp ogt <8 x float> zeroinitializer, %17
  %22 = fcmp ogt <8 x float> zeroinitializer, %18
  %23 = fcmp ogt <8 x float> zeroinitializer, %19
  ret void
}
