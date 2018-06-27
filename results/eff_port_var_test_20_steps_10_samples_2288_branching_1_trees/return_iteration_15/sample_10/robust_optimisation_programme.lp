\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj: wcvar
Subject To
 w_0_ADM:                        w_0_ADM - 0.99 b_0_ADM + 1.01 s_0_ADM  = 0.1
 w_0_AXP:                        w_0_AXP - 0.99 b_0_AXP + 1.01 s_0_AXP  = 0.1
 w_0_DIS:                        w_0_DIS - 0.99 b_0_DIS + 1.01 s_0_DIS  = 0.1
 w_0_FDX:                        w_0_FDX - 0.99 b_0_FDX + 1.01 s_0_FDX  = 0.1
 w_0_IBM:                        w_0_IBM - 0.99 b_0_IBM + 1.01 s_0_IBM  = 0.1
 w_0_INTC:                       w_0_INTC - 0.99 b_0_INTC + 1.01 s_0_INTC
                                  = 0.1
 w_0_KO:                         w_0_KO - 0.99 b_0_KO + 1.01 s_0_KO  = 0.1
 w_0_MAT:                        w_0_MAT - 0.99 b_0_MAT + 1.01 s_0_MAT  = 0.1
 w_0_MSFT:                       w_0_MSFT - 0.99 b_0_MSFT + 1.01 s_0_MSFT
                                  = 0.1
 w_0_PG:                         w_0_PG - 0.99 b_0_PG + 1.01 s_0_PG  = 0.1
 w_0_balance_constraint:         b_0_ADM + b_0_AXP + b_0_DIS + b_0_FDX
                                 + b_0_IBM + b_0_INTC + b_0_KO + b_0_MAT
                                 + b_0_MSFT + b_0_PG - s_0_ADM - s_0_AXP
                                 - s_0_DIS - s_0_FDX - s_0_IBM - s_0_INTC
                                 - s_0_KO - s_0_MAT - s_0_MSFT - s_0_PG  = 0
 w_k1_ADM_s0_return_constraint:  1.19782 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.982725 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.1679 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.830267 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.0838 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.830848 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.926298 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.853414 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.12335 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.27244 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.737466 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.01136 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.809251 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.0417 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.993125 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.871447 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.14183 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.2142 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.15649 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.34029 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.929243 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.891258 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.913446 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.929113 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.737792 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.967518 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.15081 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.32534 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.876772 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.97523 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.867355 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.719504 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.04017 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.861914 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.08168 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.10227 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.19966 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.874768 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.18962 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.924268 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.37828 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.842704 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.17826 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.781441 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.751392 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.883379 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.33531 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.17224 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.777822 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.87533 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.861702 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.16934 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.924963 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.772832 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.22566 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.16921 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.05445 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.60041 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.885057 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.914531 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.26568 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.925144 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.699556 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.726816 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.23965 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.6009 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.755469 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.976132 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.756741 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.634908 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.04917 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.01089 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.06296 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.151 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.33802 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.877318 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.34131 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.891424 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.31459 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.797759 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.24523 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.742942 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.6791 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.932807 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.61633 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.36342 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.724193 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.06162 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.745058 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.847874 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.863305 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.613576 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.939569 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.42256 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.00334 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.60677 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.86094 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.177 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.07146 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.770444 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.606919 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.979146 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.29918 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.65202 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.789735 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.991582 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.64126 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.563825 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.03721 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.768314 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.12675 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.35379 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.37702 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.815366 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.23423 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.924069 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.1507 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.841443 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.11917 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.824952 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.898836 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.960677 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.39656 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.06424 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.794021 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.12201 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.778996 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.1452 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.0575 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.750814 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.08508 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.26505 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.923639 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.34793 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.923268 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.063 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.09326 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.876205 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.783198 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.00863 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.06583 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.43202 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.737052 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.0784 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.805002 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.637473 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.08881 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.844153 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.18371 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.10266 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.31986 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.909578 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.10186 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.988504 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.15481 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.863617 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.17563 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.83472 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.608399 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.975124 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.21698 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.848 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.823307 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.779403 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.2344 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.65186 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.512614 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.913555 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.878159 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.12862 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.51116 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.62847 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.699524 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.975941 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.16852 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.00824 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.624806 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.750748 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.68769 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.1928 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.39431 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.765579 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.596764 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.830445 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.722427 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.50978 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.876588 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.51664 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.946056 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.851399 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.14984 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.914499 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.17347 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.948128 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.1938 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.884054 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.727482 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.878355 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.15551 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.02652 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.729998 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.929774 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.699849 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.11269 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.984756 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.879851 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.01298 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.11975 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.875055 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.20934 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.850033 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.979529 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.03341 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.874871 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.731558 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.923558 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.08678 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.35348 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.741502 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.940281 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.771057 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.696037 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.975629 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.812177 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.17986 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.12038 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.13558 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.996116 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.15001 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.92188 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.10147 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.814031 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.24982 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.816826 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.76101 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.03387 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.46819 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.64817 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.840137 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.07705 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.982706 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.789495 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.865604 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.849338 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.06917 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.35282 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.32175 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.6456 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.932242 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.1745 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.05558 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.864141 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.701206 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.919545 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.41034 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.29457 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.947142 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.930302 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.704364 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.698619 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.910441 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.00481 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.00438 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.41682 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.08588 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.934129 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.18949 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.940557 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.10008 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.782827 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.12752 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.771085 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.68689 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.968978 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.4914 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 2.18294 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.937382 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.912462 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.5057 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.586077 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.636998 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.899386 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.930955 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.31351 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.55184 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.60603 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.950252 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.930689 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.53146 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.13085 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.77484 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.669384 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.828 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.74434 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.40734 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.876098 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.698008 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.843359 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.817428 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.63017 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.834612 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.43385 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.05634 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.11351 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.16748 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.916328 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.1789 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.964653 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.08469 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.792704 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.800049 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.982111 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.29495 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.06175 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.779813 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.16049 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.747867 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.944295 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.895358 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.711083 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.966911 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.17501 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.889192 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.19395 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.86373 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.971447 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.18603 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.909838 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.817405 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.14688 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.13592 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.43174 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.815327 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.07426 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.865742 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.779412 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.12113 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.868216 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.26742 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.18339 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.42584 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.04107 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.11165 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.97714 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.07533 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.858488 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.06464 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.85399 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.939724 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.1426 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.31773 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.24421 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.868495 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.02219 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.941476 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.947495 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.02316 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.877254 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.12641 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.21434 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.1434 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.38056 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.975604 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.09629 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.10898 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.90393 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.794801 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.906373 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.17774 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.32587 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.889306 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.993695 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.869356 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.827726 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.940413 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.989057 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.11653 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.1684 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.14691 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.925254 w_k1_PG_s11 - w_k1_PG_s117
                                 + 0.99 b_k1_PG_s117 - 1.01 s_k1_PG_s117  = 0
 k1_s0_balance_constraint:       b_k1_ADM_s0 + b_k1_AXP_s0 + b_k1_DIS_s0
                                 + b_k1_FDX_s0 + b_k1_IBM_s0 + b_k1_INTC_s0
                                 + b_k1_KO_s0 + b_k1_MAT_s0 + b_k1_MSFT_s0
                                 + b_k1_PG_s0 - s_k1_ADM_s0 - s_k1_AXP_s0
                                 - s_k1_DIS_s0 - s_k1_FDX_s0 - s_k1_IBM_s0
                                 - s_k1_INTC_s0 - s_k1_KO_s0 - s_k1_MAT_s0
                                 - s_k1_MSFT_s0 - s_k1_PG_s0  = 0
 k1_s1_balance_constraint:       b_k1_ADM_s1 + b_k1_AXP_s1 + b_k1_DIS_s1
                                 + b_k1_FDX_s1 + b_k1_IBM_s1 + b_k1_INTC_s1
                                 + b_k1_KO_s1 + b_k1_MAT_s1 + b_k1_MSFT_s1
                                 + b_k1_PG_s1 - s_k1_ADM_s1 - s_k1_AXP_s1
                                 - s_k1_DIS_s1 - s_k1_FDX_s1 - s_k1_IBM_s1
                                 - s_k1_INTC_s1 - s_k1_KO_s1 - s_k1_MAT_s1
                                 - s_k1_MSFT_s1 - s_k1_PG_s1  = 0
 k1_s00_balance_constraint:      b_k1_ADM_s00 + b_k1_AXP_s00 + b_k1_DIS_s00
                                 + b_k1_FDX_s00 + b_k1_IBM_s00 + b_k1_INTC_s00
                                 + b_k1_KO_s00 + b_k1_MAT_s00 + b_k1_MSFT_s00
                                 + b_k1_PG_s00 - s_k1_ADM_s00 - s_k1_AXP_s00
                                 - s_k1_DIS_s00 - s_k1_FDX_s00 - s_k1_IBM_s00
                                 - s_k1_INTC_s00 - s_k1_KO_s00 - s_k1_MAT_s00
                                 - s_k1_MSFT_s00 - s_k1_PG_s00  = 0
 k1_s01_balance_constraint:      b_k1_ADM_s01 + b_k1_AXP_s01 + b_k1_DIS_s01
                                 + b_k1_FDX_s01 + b_k1_IBM_s01 + b_k1_INTC_s01
                                 + b_k1_KO_s01 + b_k1_MAT_s01 + b_k1_MSFT_s01
                                 + b_k1_PG_s01 - s_k1_ADM_s01 - s_k1_AXP_s01
                                 - s_k1_DIS_s01 - s_k1_FDX_s01 - s_k1_IBM_s01
                                 - s_k1_INTC_s01 - s_k1_KO_s01 - s_k1_MAT_s01
                                 - s_k1_MSFT_s01 - s_k1_PG_s01  = 0
 k1_s10_balance_constraint:      b_k1_ADM_s10 + b_k1_AXP_s10 + b_k1_DIS_s10
                                 + b_k1_FDX_s10 + b_k1_IBM_s10 + b_k1_INTC_s10
                                 + b_k1_KO_s10 + b_k1_MAT_s10 + b_k1_MSFT_s10
                                 + b_k1_PG_s10 - s_k1_ADM_s10 - s_k1_AXP_s10
                                 - s_k1_DIS_s10 - s_k1_FDX_s10 - s_k1_IBM_s10
                                 - s_k1_INTC_s10 - s_k1_KO_s10 - s_k1_MAT_s10
                                 - s_k1_MSFT_s10 - s_k1_PG_s10  = 0
 k1_s11_balance_constraint:      b_k1_ADM_s11 + b_k1_AXP_s11 + b_k1_DIS_s11
                                 + b_k1_FDX_s11 + b_k1_IBM_s11 + b_k1_INTC_s11
                                 + b_k1_KO_s11 + b_k1_MAT_s11 + b_k1_MSFT_s11
                                 + b_k1_PG_s11 - s_k1_ADM_s11 - s_k1_AXP_s11
                                 - s_k1_DIS_s11 - s_k1_FDX_s11 - s_k1_IBM_s11
                                 - s_k1_INTC_s11 - s_k1_KO_s11 - s_k1_MAT_s11
                                 - s_k1_MSFT_s11 - s_k1_PG_s11  = 0
 k1_s000_balance_constraint:     b_k1_ADM_s000 + b_k1_AXP_s000 + b_k1_DIS_s000
                                 + b_k1_FDX_s000 + b_k1_IBM_s000
                                 + b_k1_INTC_s000 + b_k1_KO_s000
                                 + b_k1_MAT_s000 + b_k1_MSFT_s000
                                 + b_k1_PG_s000 - s_k1_ADM_s000 - s_k1_AXP_s000
                                 - s_k1_DIS_s000 - s_k1_FDX_s000
                                 - s_k1_IBM_s000 - s_k1_INTC_s000
                                 - s_k1_KO_s000 - s_k1_MAT_s000
                                 - s_k1_MSFT_s000 - s_k1_PG_s000  = 0
 k1_s001_balance_constraint:     b_k1_ADM_s001 + b_k1_AXP_s001 + b_k1_DIS_s001
                                 + b_k1_FDX_s001 + b_k1_IBM_s001
                                 + b_k1_INTC_s001 + b_k1_KO_s001
                                 + b_k1_MAT_s001 + b_k1_MSFT_s001
                                 + b_k1_PG_s001 - s_k1_ADM_s001 - s_k1_AXP_s001
                                 - s_k1_DIS_s001 - s_k1_FDX_s001
                                 - s_k1_IBM_s001 - s_k1_INTC_s001
                                 - s_k1_KO_s001 - s_k1_MAT_s001
                                 - s_k1_MSFT_s001 - s_k1_PG_s001  = 0
 k1_s002_balance_constraint:     b_k1_ADM_s002 + b_k1_AXP_s002 + b_k1_DIS_s002
                                 + b_k1_FDX_s002 + b_k1_IBM_s002
                                 + b_k1_INTC_s002 + b_k1_KO_s002
                                 + b_k1_MAT_s002 + b_k1_MSFT_s002
                                 + b_k1_PG_s002 - s_k1_ADM_s002 - s_k1_AXP_s002
                                 - s_k1_DIS_s002 - s_k1_FDX_s002
                                 - s_k1_IBM_s002 - s_k1_INTC_s002
                                 - s_k1_KO_s002 - s_k1_MAT_s002
                                 - s_k1_MSFT_s002 - s_k1_PG_s002  = 0
 k1_s003_balance_constraint:     b_k1_ADM_s003 + b_k1_AXP_s003 + b_k1_DIS_s003
                                 + b_k1_FDX_s003 + b_k1_IBM_s003
                                 + b_k1_INTC_s003 + b_k1_KO_s003
                                 + b_k1_MAT_s003 + b_k1_MSFT_s003
                                 + b_k1_PG_s003 - s_k1_ADM_s003 - s_k1_AXP_s003
                                 - s_k1_DIS_s003 - s_k1_FDX_s003
                                 - s_k1_IBM_s003 - s_k1_INTC_s003
                                 - s_k1_KO_s003 - s_k1_MAT_s003
                                 - s_k1_MSFT_s003 - s_k1_PG_s003  = 0
 k1_s004_balance_constraint:     b_k1_ADM_s004 + b_k1_AXP_s004 + b_k1_DIS_s004
                                 + b_k1_FDX_s004 + b_k1_IBM_s004
                                 + b_k1_INTC_s004 + b_k1_KO_s004
                                 + b_k1_MAT_s004 + b_k1_MSFT_s004
                                 + b_k1_PG_s004 - s_k1_ADM_s004 - s_k1_AXP_s004
                                 - s_k1_DIS_s004 - s_k1_FDX_s004
                                 - s_k1_IBM_s004 - s_k1_INTC_s004
                                 - s_k1_KO_s004 - s_k1_MAT_s004
                                 - s_k1_MSFT_s004 - s_k1_PG_s004  = 0
 k1_s005_balance_constraint:     b_k1_ADM_s005 + b_k1_AXP_s005 + b_k1_DIS_s005
                                 + b_k1_FDX_s005 + b_k1_IBM_s005
                                 + b_k1_INTC_s005 + b_k1_KO_s005
                                 + b_k1_MAT_s005 + b_k1_MSFT_s005
                                 + b_k1_PG_s005 - s_k1_ADM_s005 - s_k1_AXP_s005
                                 - s_k1_DIS_s005 - s_k1_FDX_s005
                                 - s_k1_IBM_s005 - s_k1_INTC_s005
                                 - s_k1_KO_s005 - s_k1_MAT_s005
                                 - s_k1_MSFT_s005 - s_k1_PG_s005  = 0
 k1_s006_balance_constraint:     b_k1_ADM_s006 + b_k1_AXP_s006 + b_k1_DIS_s006
                                 + b_k1_FDX_s006 + b_k1_IBM_s006
                                 + b_k1_INTC_s006 + b_k1_KO_s006
                                 + b_k1_MAT_s006 + b_k1_MSFT_s006
                                 + b_k1_PG_s006 - s_k1_ADM_s006 - s_k1_AXP_s006
                                 - s_k1_DIS_s006 - s_k1_FDX_s006
                                 - s_k1_IBM_s006 - s_k1_INTC_s006
                                 - s_k1_KO_s006 - s_k1_MAT_s006
                                 - s_k1_MSFT_s006 - s_k1_PG_s006  = 0
 k1_s007_balance_constraint:     b_k1_ADM_s007 + b_k1_AXP_s007 + b_k1_DIS_s007
                                 + b_k1_FDX_s007 + b_k1_IBM_s007
                                 + b_k1_INTC_s007 + b_k1_KO_s007
                                 + b_k1_MAT_s007 + b_k1_MSFT_s007
                                 + b_k1_PG_s007 - s_k1_ADM_s007 - s_k1_AXP_s007
                                 - s_k1_DIS_s007 - s_k1_FDX_s007
                                 - s_k1_IBM_s007 - s_k1_INTC_s007
                                 - s_k1_KO_s007 - s_k1_MAT_s007
                                 - s_k1_MSFT_s007 - s_k1_PG_s007  = 0
 k1_s010_balance_constraint:     b_k1_ADM_s010 + b_k1_AXP_s010 + b_k1_DIS_s010
                                 + b_k1_FDX_s010 + b_k1_IBM_s010
                                 + b_k1_INTC_s010 + b_k1_KO_s010
                                 + b_k1_MAT_s010 + b_k1_MSFT_s010
                                 + b_k1_PG_s010 - s_k1_ADM_s010 - s_k1_AXP_s010
                                 - s_k1_DIS_s010 - s_k1_FDX_s010
                                 - s_k1_IBM_s010 - s_k1_INTC_s010
                                 - s_k1_KO_s010 - s_k1_MAT_s010
                                 - s_k1_MSFT_s010 - s_k1_PG_s010  = 0
 k1_s011_balance_constraint:     b_k1_ADM_s011 + b_k1_AXP_s011 + b_k1_DIS_s011
                                 + b_k1_FDX_s011 + b_k1_IBM_s011
                                 + b_k1_INTC_s011 + b_k1_KO_s011
                                 + b_k1_MAT_s011 + b_k1_MSFT_s011
                                 + b_k1_PG_s011 - s_k1_ADM_s011 - s_k1_AXP_s011
                                 - s_k1_DIS_s011 - s_k1_FDX_s011
                                 - s_k1_IBM_s011 - s_k1_INTC_s011
                                 - s_k1_KO_s011 - s_k1_MAT_s011
                                 - s_k1_MSFT_s011 - s_k1_PG_s011  = 0
 k1_s012_balance_constraint:     b_k1_ADM_s012 + b_k1_AXP_s012 + b_k1_DIS_s012
                                 + b_k1_FDX_s012 + b_k1_IBM_s012
                                 + b_k1_INTC_s012 + b_k1_KO_s012
                                 + b_k1_MAT_s012 + b_k1_MSFT_s012
                                 + b_k1_PG_s012 - s_k1_ADM_s012 - s_k1_AXP_s012
                                 - s_k1_DIS_s012 - s_k1_FDX_s012
                                 - s_k1_IBM_s012 - s_k1_INTC_s012
                                 - s_k1_KO_s012 - s_k1_MAT_s012
                                 - s_k1_MSFT_s012 - s_k1_PG_s012  = 0
 k1_s013_balance_constraint:     b_k1_ADM_s013 + b_k1_AXP_s013 + b_k1_DIS_s013
                                 + b_k1_FDX_s013 + b_k1_IBM_s013
                                 + b_k1_INTC_s013 + b_k1_KO_s013
                                 + b_k1_MAT_s013 + b_k1_MSFT_s013
                                 + b_k1_PG_s013 - s_k1_ADM_s013 - s_k1_AXP_s013
                                 - s_k1_DIS_s013 - s_k1_FDX_s013
                                 - s_k1_IBM_s013 - s_k1_INTC_s013
                                 - s_k1_KO_s013 - s_k1_MAT_s013
                                 - s_k1_MSFT_s013 - s_k1_PG_s013  = 0
 k1_s014_balance_constraint:     b_k1_ADM_s014 + b_k1_AXP_s014 + b_k1_DIS_s014
                                 + b_k1_FDX_s014 + b_k1_IBM_s014
                                 + b_k1_INTC_s014 + b_k1_KO_s014
                                 + b_k1_MAT_s014 + b_k1_MSFT_s014
                                 + b_k1_PG_s014 - s_k1_ADM_s014 - s_k1_AXP_s014
                                 - s_k1_DIS_s014 - s_k1_FDX_s014
                                 - s_k1_IBM_s014 - s_k1_INTC_s014
                                 - s_k1_KO_s014 - s_k1_MAT_s014
                                 - s_k1_MSFT_s014 - s_k1_PG_s014  = 0
 k1_s015_balance_constraint:     b_k1_ADM_s015 + b_k1_AXP_s015 + b_k1_DIS_s015
                                 + b_k1_FDX_s015 + b_k1_IBM_s015
                                 + b_k1_INTC_s015 + b_k1_KO_s015
                                 + b_k1_MAT_s015 + b_k1_MSFT_s015
                                 + b_k1_PG_s015 - s_k1_ADM_s015 - s_k1_AXP_s015
                                 - s_k1_DIS_s015 - s_k1_FDX_s015
                                 - s_k1_IBM_s015 - s_k1_INTC_s015
                                 - s_k1_KO_s015 - s_k1_MAT_s015
                                 - s_k1_MSFT_s015 - s_k1_PG_s015  = 0
 k1_s016_balance_constraint:     b_k1_ADM_s016 + b_k1_AXP_s016 + b_k1_DIS_s016
                                 + b_k1_FDX_s016 + b_k1_IBM_s016
                                 + b_k1_INTC_s016 + b_k1_KO_s016
                                 + b_k1_MAT_s016 + b_k1_MSFT_s016
                                 + b_k1_PG_s016 - s_k1_ADM_s016 - s_k1_AXP_s016
                                 - s_k1_DIS_s016 - s_k1_FDX_s016
                                 - s_k1_IBM_s016 - s_k1_INTC_s016
                                 - s_k1_KO_s016 - s_k1_MAT_s016
                                 - s_k1_MSFT_s016 - s_k1_PG_s016  = 0
 k1_s017_balance_constraint:     b_k1_ADM_s017 + b_k1_AXP_s017 + b_k1_DIS_s017
                                 + b_k1_FDX_s017 + b_k1_IBM_s017
                                 + b_k1_INTC_s017 + b_k1_KO_s017
                                 + b_k1_MAT_s017 + b_k1_MSFT_s017
                                 + b_k1_PG_s017 - s_k1_ADM_s017 - s_k1_AXP_s017
                                 - s_k1_DIS_s017 - s_k1_FDX_s017
                                 - s_k1_IBM_s017 - s_k1_INTC_s017
                                 - s_k1_KO_s017 - s_k1_MAT_s017
                                 - s_k1_MSFT_s017 - s_k1_PG_s017  = 0
 k1_s100_balance_constraint:     b_k1_ADM_s100 + b_k1_AXP_s100 + b_k1_DIS_s100
                                 + b_k1_FDX_s100 + b_k1_IBM_s100
                                 + b_k1_INTC_s100 + b_k1_KO_s100
                                 + b_k1_MAT_s100 + b_k1_MSFT_s100
                                 + b_k1_PG_s100 - s_k1_ADM_s100 - s_k1_AXP_s100
                                 - s_k1_DIS_s100 - s_k1_FDX_s100
                                 - s_k1_IBM_s100 - s_k1_INTC_s100
                                 - s_k1_KO_s100 - s_k1_MAT_s100
                                 - s_k1_MSFT_s100 - s_k1_PG_s100  = 0
 k1_s101_balance_constraint:     b_k1_ADM_s101 + b_k1_AXP_s101 + b_k1_DIS_s101
                                 + b_k1_FDX_s101 + b_k1_IBM_s101
                                 + b_k1_INTC_s101 + b_k1_KO_s101
                                 + b_k1_MAT_s101 + b_k1_MSFT_s101
                                 + b_k1_PG_s101 - s_k1_ADM_s101 - s_k1_AXP_s101
                                 - s_k1_DIS_s101 - s_k1_FDX_s101
                                 - s_k1_IBM_s101 - s_k1_INTC_s101
                                 - s_k1_KO_s101 - s_k1_MAT_s101
                                 - s_k1_MSFT_s101 - s_k1_PG_s101  = 0
 k1_s102_balance_constraint:     b_k1_ADM_s102 + b_k1_AXP_s102 + b_k1_DIS_s102
                                 + b_k1_FDX_s102 + b_k1_IBM_s102
                                 + b_k1_INTC_s102 + b_k1_KO_s102
                                 + b_k1_MAT_s102 + b_k1_MSFT_s102
                                 + b_k1_PG_s102 - s_k1_ADM_s102 - s_k1_AXP_s102
                                 - s_k1_DIS_s102 - s_k1_FDX_s102
                                 - s_k1_IBM_s102 - s_k1_INTC_s102
                                 - s_k1_KO_s102 - s_k1_MAT_s102
                                 - s_k1_MSFT_s102 - s_k1_PG_s102  = 0
 k1_s103_balance_constraint:     b_k1_ADM_s103 + b_k1_AXP_s103 + b_k1_DIS_s103
                                 + b_k1_FDX_s103 + b_k1_IBM_s103
                                 + b_k1_INTC_s103 + b_k1_KO_s103
                                 + b_k1_MAT_s103 + b_k1_MSFT_s103
                                 + b_k1_PG_s103 - s_k1_ADM_s103 - s_k1_AXP_s103
                                 - s_k1_DIS_s103 - s_k1_FDX_s103
                                 - s_k1_IBM_s103 - s_k1_INTC_s103
                                 - s_k1_KO_s103 - s_k1_MAT_s103
                                 - s_k1_MSFT_s103 - s_k1_PG_s103  = 0
 k1_s104_balance_constraint:     b_k1_ADM_s104 + b_k1_AXP_s104 + b_k1_DIS_s104
                                 + b_k1_FDX_s104 + b_k1_IBM_s104
                                 + b_k1_INTC_s104 + b_k1_KO_s104
                                 + b_k1_MAT_s104 + b_k1_MSFT_s104
                                 + b_k1_PG_s104 - s_k1_ADM_s104 - s_k1_AXP_s104
                                 - s_k1_DIS_s104 - s_k1_FDX_s104
                                 - s_k1_IBM_s104 - s_k1_INTC_s104
                                 - s_k1_KO_s104 - s_k1_MAT_s104
                                 - s_k1_MSFT_s104 - s_k1_PG_s104  = 0
 k1_s105_balance_constraint:     b_k1_ADM_s105 + b_k1_AXP_s105 + b_k1_DIS_s105
                                 + b_k1_FDX_s105 + b_k1_IBM_s105
                                 + b_k1_INTC_s105 + b_k1_KO_s105
                                 + b_k1_MAT_s105 + b_k1_MSFT_s105
                                 + b_k1_PG_s105 - s_k1_ADM_s105 - s_k1_AXP_s105
                                 - s_k1_DIS_s105 - s_k1_FDX_s105
                                 - s_k1_IBM_s105 - s_k1_INTC_s105
                                 - s_k1_KO_s105 - s_k1_MAT_s105
                                 - s_k1_MSFT_s105 - s_k1_PG_s105  = 0
 k1_s106_balance_constraint:     b_k1_ADM_s106 + b_k1_AXP_s106 + b_k1_DIS_s106
                                 + b_k1_FDX_s106 + b_k1_IBM_s106
                                 + b_k1_INTC_s106 + b_k1_KO_s106
                                 + b_k1_MAT_s106 + b_k1_MSFT_s106
                                 + b_k1_PG_s106 - s_k1_ADM_s106 - s_k1_AXP_s106
                                 - s_k1_DIS_s106 - s_k1_FDX_s106
                                 - s_k1_IBM_s106 - s_k1_INTC_s106
                                 - s_k1_KO_s106 - s_k1_MAT_s106
                                 - s_k1_MSFT_s106 - s_k1_PG_s106  = 0
 k1_s107_balance_constraint:     b_k1_ADM_s107 + b_k1_AXP_s107 + b_k1_DIS_s107
                                 + b_k1_FDX_s107 + b_k1_IBM_s107
                                 + b_k1_INTC_s107 + b_k1_KO_s107
                                 + b_k1_MAT_s107 + b_k1_MSFT_s107
                                 + b_k1_PG_s107 - s_k1_ADM_s107 - s_k1_AXP_s107
                                 - s_k1_DIS_s107 - s_k1_FDX_s107
                                 - s_k1_IBM_s107 - s_k1_INTC_s107
                                 - s_k1_KO_s107 - s_k1_MAT_s107
                                 - s_k1_MSFT_s107 - s_k1_PG_s107  = 0
 k1_s110_balance_constraint:     b_k1_ADM_s110 + b_k1_AXP_s110 + b_k1_DIS_s110
                                 + b_k1_FDX_s110 + b_k1_IBM_s110
                                 + b_k1_INTC_s110 + b_k1_KO_s110
                                 + b_k1_MAT_s110 + b_k1_MSFT_s110
                                 + b_k1_PG_s110 - s_k1_ADM_s110 - s_k1_AXP_s110
                                 - s_k1_DIS_s110 - s_k1_FDX_s110
                                 - s_k1_IBM_s110 - s_k1_INTC_s110
                                 - s_k1_KO_s110 - s_k1_MAT_s110
                                 - s_k1_MSFT_s110 - s_k1_PG_s110  = 0
 k1_s111_balance_constraint:     b_k1_ADM_s111 + b_k1_AXP_s111 + b_k1_DIS_s111
                                 + b_k1_FDX_s111 + b_k1_IBM_s111
                                 + b_k1_INTC_s111 + b_k1_KO_s111
                                 + b_k1_MAT_s111 + b_k1_MSFT_s111
                                 + b_k1_PG_s111 - s_k1_ADM_s111 - s_k1_AXP_s111
                                 - s_k1_DIS_s111 - s_k1_FDX_s111
                                 - s_k1_IBM_s111 - s_k1_INTC_s111
                                 - s_k1_KO_s111 - s_k1_MAT_s111
                                 - s_k1_MSFT_s111 - s_k1_PG_s111  = 0
 k1_s112_balance_constraint:     b_k1_ADM_s112 + b_k1_AXP_s112 + b_k1_DIS_s112
                                 + b_k1_FDX_s112 + b_k1_IBM_s112
                                 + b_k1_INTC_s112 + b_k1_KO_s112
                                 + b_k1_MAT_s112 + b_k1_MSFT_s112
                                 + b_k1_PG_s112 - s_k1_ADM_s112 - s_k1_AXP_s112
                                 - s_k1_DIS_s112 - s_k1_FDX_s112
                                 - s_k1_IBM_s112 - s_k1_INTC_s112
                                 - s_k1_KO_s112 - s_k1_MAT_s112
                                 - s_k1_MSFT_s112 - s_k1_PG_s112  = 0
 k1_s113_balance_constraint:     b_k1_ADM_s113 + b_k1_AXP_s113 + b_k1_DIS_s113
                                 + b_k1_FDX_s113 + b_k1_IBM_s113
                                 + b_k1_INTC_s113 + b_k1_KO_s113
                                 + b_k1_MAT_s113 + b_k1_MSFT_s113
                                 + b_k1_PG_s113 - s_k1_ADM_s113 - s_k1_AXP_s113
                                 - s_k1_DIS_s113 - s_k1_FDX_s113
                                 - s_k1_IBM_s113 - s_k1_INTC_s113
                                 - s_k1_KO_s113 - s_k1_MAT_s113
                                 - s_k1_MSFT_s113 - s_k1_PG_s113  = 0
 k1_s114_balance_constraint:     b_k1_ADM_s114 + b_k1_AXP_s114 + b_k1_DIS_s114
                                 + b_k1_FDX_s114 + b_k1_IBM_s114
                                 + b_k1_INTC_s114 + b_k1_KO_s114
                                 + b_k1_MAT_s114 + b_k1_MSFT_s114
                                 + b_k1_PG_s114 - s_k1_ADM_s114 - s_k1_AXP_s114
                                 - s_k1_DIS_s114 - s_k1_FDX_s114
                                 - s_k1_IBM_s114 - s_k1_INTC_s114
                                 - s_k1_KO_s114 - s_k1_MAT_s114
                                 - s_k1_MSFT_s114 - s_k1_PG_s114  = 0
 k1_s115_balance_constraint:     b_k1_ADM_s115 + b_k1_AXP_s115 + b_k1_DIS_s115
                                 + b_k1_FDX_s115 + b_k1_IBM_s115
                                 + b_k1_INTC_s115 + b_k1_KO_s115
                                 + b_k1_MAT_s115 + b_k1_MSFT_s115
                                 + b_k1_PG_s115 - s_k1_ADM_s115 - s_k1_AXP_s115
                                 - s_k1_DIS_s115 - s_k1_FDX_s115
                                 - s_k1_IBM_s115 - s_k1_INTC_s115
                                 - s_k1_KO_s115 - s_k1_MAT_s115
                                 - s_k1_MSFT_s115 - s_k1_PG_s115  = 0
 k1_s116_balance_constraint:     b_k1_ADM_s116 + b_k1_AXP_s116 + b_k1_DIS_s116
                                 + b_k1_FDX_s116 + b_k1_IBM_s116
                                 + b_k1_INTC_s116 + b_k1_KO_s116
                                 + b_k1_MAT_s116 + b_k1_MSFT_s116
                                 + b_k1_PG_s116 - s_k1_ADM_s116 - s_k1_AXP_s116
                                 - s_k1_DIS_s116 - s_k1_FDX_s116
                                 - s_k1_IBM_s116 - s_k1_INTC_s116
                                 - s_k1_KO_s116 - s_k1_MAT_s116
                                 - s_k1_MSFT_s116 - s_k1_PG_s116  = 0
 k1_s117_balance_constraint:     b_k1_ADM_s117 + b_k1_AXP_s117 + b_k1_DIS_s117
                                 + b_k1_FDX_s117 + b_k1_IBM_s117
                                 + b_k1_INTC_s117 + b_k1_KO_s117
                                 + b_k1_MAT_s117 + b_k1_MSFT_s117
                                 + b_k1_PG_s117 - s_k1_ADM_s117 - s_k1_AXP_s117
                                 - s_k1_DIS_s117 - s_k1_FDX_s117
                                 - s_k1_IBM_s117 - s_k1_INTC_s117
                                 - s_k1_KO_s117 - s_k1_MAT_s117
                                 - s_k1_MSFT_s117 - s_k1_PG_s117  = 0
 k1_wcvar_constraint:            wcvar - var_1 - 0.198999708971228 z_k1_s0000
                                 - 0.202999662774743 z_k1_s0001
                                 - 0.225998583396414 z_k1_s0002
                                 - 0.403999023148314 z_k1_s0003
                                 - 0.439999258378772 z_k1_s0004
                                 - 0.372000260718637 z_k1_s0005
                                 - 0.0899999370773164 z_k1_s0006
                                 - 0.236000095402272 z_k1_s0007
                                 - 0.272999597553877 z_k1_s0010
                                 - 0.580000338720894 z_k1_s0011
                                 - 0.438998791085572 z_k1_s0012
                                 - 0.344998824327238 z_k1_s0013
                                 - 0.458999463799184 z_k1_s0014
                                 - 0.27799976573228 z_k1_s0015
                                 - 0.446999060171017 z_k1_s0016
                                 - 0.373999799761975 z_k1_s0017
                                 - 0.295001441832899 z_k1_s0020
                                 - 0.260000721105348 z_k1_s0021
                                 - 0.221000288638594 z_k1_s0022
                                 - 0.316999856398519 z_k1_s0023
                                 - 0.328000144684502 z_k1_s0024
                                 - 0.280001441807712 z_k1_s0025
                                 - 0.192000648924291 z_k1_s0026
                                 - 0.269001153521729 z_k1_s0027
                                 - 0.538001676908431 z_k1_s0030
                                 - 0.208000252018014 z_k1_s0031
                                 - 0.232999800228722 z_k1_s0032
                                 - 0.122000117371106 z_k1_s0033
                                 - 0.134000111612505 z_k1_s0034
                                 - 0.181000317749931 z_k1_s0035
                                 - 0.18200029967836 z_k1_s0036
                                 - 0.513001073196427 z_k1_s0037
                                 - 0.188999554820931 z_k1_s0040
                                 - 0.155999588010922 z_k1_s0041
                                 - 0.0589999449118346 z_k1_s0042
                                 - 0.176999694435487 z_k1_s0043
                                 - 0.23300044590105 z_k1_s0044
                                 - 0.266000412711059 z_k1_s0045
                                 - 0.251999874094628 z_k1_s0046
                                 - 0.0709999455972938 z_k1_s0047
                                 - 0.172000297559444 z_k1_s0050
                                 - 0.24000057440876 z_k1_s0051
                                 - 0.231000353193138 z_k1_s0052
                                 - 0.288999750357499 z_k1_s0053
                                 - 0.295001419170148 z_k1_s0054
                                 - 0.554001953477809 z_k1_s0055
                                 - 0.163000304544157 z_k1_s0056
                                 - 0.33800120830959 z_k1_s0057
                                 - 0.188000428956565 z_k1_s0060
                                 - 0.181000316694681 z_k1_s0061
                                 - 0.780002164873795 z_k1_s0062
                                 - 0.298001198426925 z_k1_s0063
                                 - 0.382999577672077 z_k1_s0064
                                 - 0.471000193248365 z_k1_s0065
                                 - 0.30900197162543 z_k1_s0066
                                 - 0.175000300041806 z_k1_s0067
                                 - 0.0460000583880113 z_k1_s0070
                                 - 0.141000051967456 z_k1_s0071
                                 - 0.0770002833584034 z_k1_s0072
                                 - 0.0380000231118069 z_k1_s0073
                                 - 0.0470000387225095 z_k1_s0074
                                 - 0.096999890048368 z_k1_s0075
                                 - 0.0490000635915788 z_k1_s0076
                                 - 0.147000190774736 z_k1_s0077
                                 - 0.106999705458709 z_k1_s0100
                                 - 0.449999458787548 z_k1_s0101
                                 - 0.134999659936625 z_k1_s0102
                                 - 0.090999807628317 z_k1_s0103
                                 - 0.413999288844977 z_k1_s0104
                                 - 0.202000065068452 z_k1_s0105
                                 - 0.126999799871249 z_k1_s0106
                                 - 0.250998963631879 z_k1_s0107
                                 - 0.344998530607557 z_k1_s0110
                                 - 0.186999369507048 z_k1_s0111
                                 - 0.136999646814444 z_k1_s0112
                                 - 0.460997494639643 z_k1_s0113
                                 - 0.252000247613504 z_k1_s0114
                                 - 0.305998798321163 z_k1_s0115
                                 - 0.270999997343153 z_k1_s0116
                                 - 0.377999202923962 z_k1_s0117
                                 - 0.265000729287814 z_k1_s0120
                                 - 0.197000669750007 z_k1_s0121
                                 - 0.230000110965972 z_k1_s0122
                                 - 0.257000631871461 z_k1_s0123
                                 - 0.223000602102558 z_k1_s0124
                                 - 0.120000231643385 z_k1_s0125
                                 - 0.0800000519620979 z_k1_s0126
                                 - 0.165000280084595 z_k1_s0127
                                 - 0.178999753729844 z_k1_s0130
                                 - 0.189999550886475 z_k1_s0131
                                 - 0.0769999800960557 z_k1_s0132
                                 - 0.258999933050369 z_k1_s0133
                                 - 0.216999943907066 z_k1_s0134
                                 - 0.185000152178784 z_k1_s0135
                                 - 0.178999753729844 z_k1_s0136
                                 - 0.114000010531812 z_k1_s0137
                                 - 0.387001600668411 z_k1_s0140
                                 - 0.365000443325308 z_k1_s0141
                                 - 0.292001349062838 z_k1_s0142
                                 - 0.389000349877886 z_k1_s0143
                                 - 0.23100071647277 z_k1_s0144
                                 - 0.261000848264139 z_k1_s0145
                                 - 0.249000894987851 z_k1_s0146
                                 - 0.312001271189986 z_k1_s0147
                                 - 0.276000631160086 z_k1_s0150
                                 - 0.22200047649832 z_k1_s0151
                                 - 0.07600001584983 z_k1_s0152
                                 - 0.14399945643108 z_k1_s0153
                                 - 0.32699942289578 z_k1_s0154
                                 - 0.206999397869733 z_k1_s0155
                                 - 0.109000094432015 z_k1_s0156
                                 - 0.073000086924201 z_k1_s0157
                                 - 0.176999539702358 z_k1_s0160
                                 - 0.286999418081876 z_k1_s0161
                                 - 0.442998500804085 z_k1_s0162
                                 - 0.319999058196671 z_k1_s0163
                                 - 0.214999360488408 z_k1_s0164
                                 - 0.151999498743575 z_k1_s0165
                                 - 0.377999601667743 z_k1_s0166
                                 - 0.183999404563243 z_k1_s0167
                                 - 0.150999757616546 z_k1_s0170
                                 - 0.25299903026761 z_k1_s0171
                                 - 0.366999183470344 z_k1_s0172
                                 - 0.158999775851814 z_k1_s0173
                                 - 0.534998286612838 z_k1_s0174
                                 - 0.121999744838619 z_k1_s0175
                                 - 0.254999781375348 z_k1_s0176
                                 - 0.291000503333132 z_k1_s0177
                                 - 0.2010001679803 z_k1_s1000
                                 - 0.275999631259155 z_k1_s1001
                                 - 0.251001380200483 z_k1_s1002
                                 - 0.180000050530059 z_k1_s1003
                                 - 0.453999375484252 z_k1_s1004
                                 - 0.142000180339354 z_k1_s1005
                                 - 0.530999640868369 z_k1_s1006
                                 - 0.443999083460186 z_k1_s1007
                                 - 0.483999598730191 z_k1_s1010
                                 - 0.630001013924619 z_k1_s1011
                                 - 0.900999915785299 z_k1_s1012
                                 - 0.985997102031101 z_k1_s1013
                                 - 0.723001871732057 z_k1_s1014
                                 - 0.700998413683047 z_k1_s1015
                                 - 0.465999569760917 z_k1_s1016
                                 - 0.419999967816894 z_k1_s1017
                                 - 0.621001414632399 z_k1_s1020
                                 - 0.60499956039018 z_k1_s1021
                                 - 0.763001152277655 z_k1_s1022
                                 - 0.899998638596775 z_k1_s1023
                                 - 1.02400097147078 z_k1_s1024
                                 - 0.846001074281593 z_k1_s1025
                                 - 0.312000312713757 z_k1_s1026
                                 - 0.278999898965085 z_k1_s1027
                                 - 0.464000432822067 z_k1_s1030
                                 - 0.224000066645063 z_k1_s1031
                                 - 0.364001139998752 z_k1_s1032
                                 - 0.383998935162366 z_k1_s1033
                                 - 0.485999727002916 z_k1_s1034
                                 - 0.625000738648337 z_k1_s1035
                                 - 0.49999990311832 z_k1_s1036
                                 - 0.39200149222956 z_k1_s1037
                                 - 1.00700064823476 z_k1_s1040
                                 - 0.584000765692799 z_k1_s1041
                                 - 0.364000457034524 z_k1_s1042
                                 - 0.424000863141777 z_k1_s1043
                                 - 1.50000129966893 z_k1_s1044
                                 - 0.548000876148942 z_k1_s1045
                                 - 0.422000672544621 z_k1_s1046
                                 - 1.05300090056357 z_k1_s1047
                                 - 0.662996788683766 z_k1_s1050
                                 - 0.355999472170925 z_k1_s1051
                                 - 0.513999120266078 z_k1_s1052
                                 - 1.62299854862263 z_k1_s1053
                                 - 0.557998944305362 z_k1_s1054
                                 - 0.667000132136388 z_k1_s1055
                                 - 1.24099951637339 z_k1_s1056
                                 - 0.536999252249883 z_k1_s1057
                                 - 0.583998354822594 z_k1_s1060
                                 - 0.71800195034801 z_k1_s1061
                                 - 0.886999275150541 z_k1_s1062
                                 - 0.178000091189236 z_k1_s1063
                                 - 0.389998303391642 z_k1_s1064
                                 - 0.539999143158054 z_k1_s1065
                                 - 0.222999955119056 z_k1_s1066
                                 - 0.360000075438874 z_k1_s1067
                                 - 2.34699871025145 z_k1_s1070
                                 - 0.67500008288182 z_k1_s1071
                                 - 0.908999789699607 z_k1_s1072
                                 - 1.29099508079818 z_k1_s1073
                                 - 0.659999482609056 z_k1_s1074
                                 - 1.77600272735414 z_k1_s1075
                                 - 0.740000207797815 z_k1_s1076
                                 - 0.887999506477583 z_k1_s1077
                                 - 0.182999803052574 z_k1_s1100
                                 - 0.275000310324113 z_k1_s1101
                                 - 0.121999794735146 z_k1_s1102
                                 - 0.27099945855687 z_k1_s1103
                                 - 0.48399877097276 z_k1_s1104
                                 - 0.181999701060618 z_k1_s1105
                                 - 0.489000390431088 z_k1_s1106
                                 - 0.21299975621531 z_k1_s1107
                                 - 0.261999482362415 z_k1_s1110
                                 - 0.554998475634733 z_k1_s1111
                                 - 0.194999761655024 z_k1_s1112
                                 - 0.13999989938043 z_k1_s1113
                                 - 0.280000221760258 z_k1_s1114
                                 - 0.112999847782162 z_k1_s1115
                                 - 0.238000611495618 z_k1_s1116
                                 - 0.33199953780248 z_k1_s1117
                                 - 0.321998957166131 z_k1_s1120
                                 - 0.527997791628158 z_k1_s1121
                                 - 0.62699865255407 z_k1_s1122
                                 - 0.238999121437331 z_k1_s1123
                                 - 0.397999236868137 z_k1_s1124
                                 - 0.212999410485327 z_k1_s1125
                                 - 0.772997552513533 z_k1_s1126
                                 - 0.299998992515152 z_k1_s1127
                                 - 0.19599987649362 z_k1_s1130
                                 - 0.406998879086069 z_k1_s1131
                                 - 0.304999081059662 z_k1_s1132
                                 - 0.217999694330824 z_k1_s1133
                                 - 0.143999878660649 z_k1_s1134
                                 - 0.252999940025851 z_k1_s1135
                                 - 0.428999339522643 z_k1_s1136
                                 - 0.189999711974593 z_k1_s1137
                                 - 0.416999005605991 z_k1_s1140
                                 - 0.105999891642592 z_k1_s1141
                                 - 0.427000369268692 z_k1_s1142
                                 - 0.0989999210774875 z_k1_s1143
                                 - 0.284999504435522 z_k1_s1144
                                 - 0.346999299954948 z_k1_s1145
                                 - 0.160999716596913 z_k1_s1146
                                 - 0.125999863771392 z_k1_s1147
                                 - 0.360000101986661 z_k1_s1150
                                 - 0.803001318600476 z_k1_s1151
                                 - 0.826998401638972 z_k1_s1152
                                 - 0.919001520956213 z_k1_s1153
                                 - 0.750998104681731 z_k1_s1154
                                 - 0.300000243888917 z_k1_s1155
                                 - 0.186000275153155 z_k1_s1156
                                 - 0.62100198738737 z_k1_s1157
                                 - 0.39099885102424 z_k1_s1160
                                 - 0.530998313060384 z_k1_s1161
                                 - 0.38599890124429 z_k1_s1162
                                 - 0.491998878416293 z_k1_s1163
                                 - 0.323999176693894 z_k1_s1164
                                 - 0.616997622915024 z_k1_s1165
                                 - 0.409998920647316 z_k1_s1166
                                 - 1.18999499320844 z_k1_s1167
                                 - 0.502999973447867 z_k1_s1170
                                 - 0.788002300842531 z_k1_s1171
                                 - 0.631999694116495 z_k1_s1172
                                 - 0.59800214444665 z_k1_s1173
                                 - 0.474000032187477 z_k1_s1174
                                 - 0.724998531509771 z_k1_s1175
                                 - 0.722999906706157 z_k1_s1176
                                 - 0.789001613244338 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.07716 w_k1_ADM_s000
                                 + 1.62574 w_k1_AXP_s000
                                 + 1.08393 w_k1_DIS_s000
                                 + 1.14271 w_k1_FDX_s000
                                 + 1.05485 w_k1_IBM_s000
                                 + 1.18693 w_k1_INTC_s000
                                 + 0.975442 w_k1_KO_s000 + 1.4069 w_k1_MAT_s000
                                 + 1.03239 w_k1_MSFT_s000
                                 + 1.02544 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.11954 w_k1_ADM_s000
                                 + 1.67994 w_k1_AXP_s000 + 1.2956 w_k1_DIS_s000
                                 + 1.01574 w_k1_FDX_s000
                                 + 1.67703 w_k1_IBM_s000
                                 + 1.43463 w_k1_INTC_s000
                                 + 1.52905 w_k1_KO_s000 + 1.79315 w_k1_MAT_s000
                                 + 1.11567 w_k1_MSFT_s000
                                 + 1.03478 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.09901 w_k1_ADM_s000
                                 + 1.26049 w_k1_AXP_s000
                                 + 1.52188 w_k1_DIS_s000
                                 + 1.25585 w_k1_FDX_s000
                                 + 1.18317 w_k1_IBM_s000
                                 + 1.44184 w_k1_INTC_s000
                                 + 1.30526 w_k1_KO_s000 + 1.3833 w_k1_MAT_s000
                                 + 1.39185 w_k1_MSFT_s000
                                 + 0.988092 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.06175 w_k1_ADM_s000
                                 + 1.11551 w_k1_AXP_s000 + 1.0801 w_k1_DIS_s000
                                 + 1.11898 w_k1_FDX_s000
                                 + 0.918949 w_k1_IBM_s000
                                 + 1.17298 w_k1_INTC_s000
                                 + 0.987367 w_k1_KO_s000
                                 + 1.04758 w_k1_MAT_s000
                                 + 1.23302 w_k1_MSFT_s000
                                 + 0.958811 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.715689 w_k1_ADM_s000
                                 + 0.781771 w_k1_AXP_s000
                                 + 0.781284 w_k1_DIS_s000
                                 + 0.7894 w_k1_FDX_s000
                                 + 0.916636 w_k1_IBM_s000
                                 + 0.981447 w_k1_INTC_s000
                                 + 0.996618 w_k1_KO_s000
                                 + 0.781486 w_k1_MAT_s000
                                 + 0.952984 w_k1_MSFT_s000
                                 + 0.825755 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.920643 w_k1_ADM_s000
                                 + 1.03263 w_k1_AXP_s000
                                 + 1.03745 w_k1_DIS_s000
                                 + 0.91384 w_k1_FDX_s000
                                 + 0.972821 w_k1_IBM_s000
                                 + 1.3956 w_k1_INTC_s000
                                 + 0.945034 w_k1_KO_s000
                                 + 0.897593 w_k1_MAT_s000
                                 + 0.92308 w_k1_MSFT_s000
                                 + 0.88278 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.04806 w_k1_ADM_s000
                                 + 1.85701 w_k1_AXP_s000
                                 + 1.89199 w_k1_DIS_s000
                                 + 1.43785 w_k1_FDX_s000
                                 + 1.27925 w_k1_IBM_s000
                                 + 1.75123 w_k1_INTC_s000
                                 + 1.43719 w_k1_KO_s000 + 1.16778 w_k1_MAT_s000
                                 + 1.39586 w_k1_MSFT_s000
                                 + 1.25581 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.788432 w_k1_ADM_s000
                                 + 0.819553 w_k1_AXP_s000
                                 + 0.618209 w_k1_DIS_s000
                                 + 0.752052 w_k1_FDX_s000
                                 + 0.621157 w_k1_IBM_s000
                                 + 1.0035 w_k1_INTC_s000
                                 + 0.737628 w_k1_KO_s000
                                 + 0.661369 w_k1_MAT_s000
                                 + 0.736748 w_k1_MSFT_s000
                                 + 0.715335 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.821541 w_k1_ADM_s001
                                 + 0.833415 w_k1_AXP_s001
                                 + 0.692954 w_k1_DIS_s001
                                 + 0.807293 w_k1_FDX_s001
                                 + 0.974032 w_k1_IBM_s001
                                 + 1.03376 w_k1_INTC_s001
                                 + 0.963133 w_k1_KO_s001
                                 + 1.34261 w_k1_MAT_s001
                                 + 0.792459 w_k1_MSFT_s001
                                 + 0.848441 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.21272 w_k1_ADM_s001
                                 + 1.32446 w_k1_AXP_s001
                                 + 1.40465 w_k1_DIS_s001
                                 + 1.20946 w_k1_FDX_s001
                                 + 1.10786 w_k1_IBM_s001
                                 + 1.37894 w_k1_INTC_s001
                                 + 1.26349 w_k1_KO_s001 + 1.27396 w_k1_MAT_s001
                                 + 1.22226 w_k1_MSFT_s001
                                 + 1.05053 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.25022 w_k1_ADM_s001
                                 + 1.05444 w_k1_AXP_s001
                                 + 1.08561 w_k1_DIS_s001
                                 + 1.05313 w_k1_FDX_s001
                                 + 0.877946 w_k1_IBM_s001
                                 + 0.898583 w_k1_INTC_s001
                                 + 1.0084 w_k1_KO_s001 + 0.80943 w_k1_MAT_s001
                                 + 0.969498 w_k1_MSFT_s001
                                 + 0.959279 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.663573 w_k1_ADM_s001
                                 + 0.55406 w_k1_AXP_s001
                                 + 0.605472 w_k1_DIS_s001
                                 + 0.711026 w_k1_FDX_s001
                                 + 0.698713 w_k1_IBM_s001
                                 + 0.811972 w_k1_INTC_s001
                                 + 0.723688 w_k1_KO_s001
                                 + 0.612702 w_k1_MAT_s001
                                 + 0.810478 w_k1_MSFT_s001
                                 + 0.635462 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.01517 w_k1_ADM_s001
                                 + 1.12535 w_k1_AXP_s001
                                 + 1.02237 w_k1_DIS_s001
                                 + 1.00771 w_k1_FDX_s001
                                 + 1.27436 w_k1_IBM_s001
                                 + 1.21847 w_k1_INTC_s001
                                 + 1.26227 w_k1_KO_s001 + 1.56083 w_k1_MAT_s001
                                 + 1.03134 w_k1_MSFT_s001
                                 + 1.03325 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.18332 w_k1_ADM_s001
                                 + 1.30489 w_k1_AXP_s001
                                 + 1.37022 w_k1_DIS_s001
                                 + 1.09884 w_k1_FDX_s001
                                 + 1.70531 w_k1_IBM_s001
                                 + 1.17861 w_k1_INTC_s001
                                 + 1.38186 w_k1_KO_s001 + 1.41051 w_k1_MAT_s001
                                 + 1.14909 w_k1_MSFT_s001
                                 + 1.05585 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.0332 w_k1_ADM_s001
                                 + 0.87518 w_k1_AXP_s001
                                 + 0.856457 w_k1_DIS_s001
                                 + 0.954808 w_k1_FDX_s001
                                 + 0.649179 w_k1_IBM_s001
                                 + 1.01659 w_k1_INTC_s001
                                 + 0.791884 w_k1_KO_s001
                                 + 0.732904 w_k1_MAT_s001
                                 + 0.850398 w_k1_MSFT_s001
                                 + 0.801813 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.970686 w_k1_ADM_s001
                                 + 0.764337 w_k1_AXP_s001
                                 + 0.904935 w_k1_DIS_s001
                                 + 0.913711 w_k1_FDX_s001
                                 + 1.04443 w_k1_IBM_s001
                                 + 1.20469 w_k1_INTC_s001
                                 + 0.99634 w_k1_KO_s001
                                 + 0.889363 w_k1_MAT_s001
                                 + 0.977973 w_k1_MSFT_s001
                                 + 0.868593 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.3189 w_k1_ADM_s002
                                 + 1.25392 w_k1_AXP_s002
                                 + 1.28815 w_k1_DIS_s002
                                 + 1.26222 w_k1_FDX_s002
                                 + 0.803359 w_k1_IBM_s002
                                 + 1.34457 w_k1_INTC_s002
                                 + 1.14181 w_k1_KO_s002
                                 + 0.668906 w_k1_MAT_s002
                                 + 1.21169 w_k1_MSFT_s002
                                 + 1.06667 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.00314 w_k1_ADM_s002
                                 + 0.596926 w_k1_AXP_s002
                                 + 0.634237 w_k1_DIS_s002
                                 + 0.804654 w_k1_FDX_s002
                                 + 0.628544 w_k1_IBM_s002
                                 + 0.761137 w_k1_INTC_s002
                                 + 0.697213 w_k1_KO_s002
                                 + 0.641954 w_k1_MAT_s002
                                 + 0.809132 w_k1_MSFT_s002
                                 + 0.766102 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.964603 w_k1_ADM_s002
                                 + 0.85668 w_k1_AXP_s002
                                 + 0.642354 w_k1_DIS_s002
                                 + 0.784537 w_k1_FDX_s002
                                 + 1.01755 w_k1_IBM_s002
                                 + 1.02158 w_k1_INTC_s002
                                 + 0.83349 w_k1_KO_s002
                                 + 0.955399 w_k1_MAT_s002
                                 + 0.856344 w_k1_MSFT_s002
                                 + 0.992257 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.20971 w_k1_ADM_s002
                                 + 0.825347 w_k1_AXP_s002
                                 + 1.00849 w_k1_DIS_s002
                                 + 0.967701 w_k1_FDX_s002
                                 + 0.919537 w_k1_IBM_s002
                                 + 0.968607 w_k1_INTC_s002
                                 + 1.04801 w_k1_KO_s002
                                 + 0.704865 w_k1_MAT_s002
                                 + 1.00773 w_k1_MSFT_s002
                                 + 0.964429 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.44881 w_k1_ADM_s002
                                 + 1.37099 w_k1_AXP_s002
                                 + 1.57196 w_k1_DIS_s002
                                 + 1.19538 w_k1_FDX_s002
                                 + 1.36946 w_k1_IBM_s002
                                 + 1.41881 w_k1_INTC_s002 + 1.2864 w_k1_KO_s002
                                 + 1.02484 w_k1_MAT_s002
                                 + 1.30903 w_k1_MSFT_s002
                                 + 1.13102 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.951914 w_k1_ADM_s002
                                 + 0.865803 w_k1_AXP_s002
                                 + 0.827843 w_k1_DIS_s002
                                 + 0.888412 w_k1_FDX_s002
                                 + 0.762233 w_k1_IBM_s002
                                 + 0.82152 w_k1_INTC_s002
                                 + 0.786404 w_k1_KO_s002
                                 + 0.848029 w_k1_MAT_s002
                                 + 0.912861 w_k1_MSFT_s002
                                 + 0.914418 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.25794 w_k1_ADM_s002
                                 + 0.959986 w_k1_AXP_s002
                                 + 0.874777 w_k1_DIS_s002
                                 + 1.09048 w_k1_FDX_s002
                                 + 0.555005 w_k1_IBM_s002
                                 + 0.935275 w_k1_INTC_s002
                                 + 0.677198 w_k1_KO_s002
                                 + 0.50234 w_k1_MAT_s002
                                 + 0.934659 w_k1_MSFT_s002
                                 + 0.796425 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.10313 w_k1_ADM_s002
                                 + 1.23505 w_k1_AXP_s002
                                 + 0.987934 w_k1_DIS_s002
                                 + 1.13862 w_k1_FDX_s002
                                 + 1.08691 w_k1_IBM_s002
                                 + 1.16865 w_k1_INTC_s002
                                 + 1.07237 w_k1_KO_s002 + 1.35568 w_k1_MAT_s002
                                 + 1.01002 w_k1_MSFT_s002 + 1.0822 w_k1_PG_s002
                                 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.762443 w_k1_ADM_s003
                                 + 1.05281 w_k1_AXP_s003
                                 + 1.06133 w_k1_DIS_s003
                                 + 1.13367 w_k1_FDX_s003
                                 + 0.883159 w_k1_IBM_s003
                                 + 1.16849 w_k1_INTC_s003
                                 + 0.907328 w_k1_KO_s003
                                 + 0.803981 w_k1_MAT_s003
                                 + 1.13545 w_k1_MSFT_s003
                                 + 0.974457 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.615492 w_k1_ADM_s003
                                 + 0.840069 w_k1_AXP_s003
                                 + 0.746048 w_k1_DIS_s003
                                 + 0.814012 w_k1_FDX_s003
                                 + 1.29026 w_k1_IBM_s003
                                 + 0.936477 w_k1_INTC_s003
                                 + 0.925947 w_k1_KO_s003
                                 + 0.879473 w_k1_MAT_s003
                                 + 0.89024 w_k1_MSFT_s003
                                 + 0.889862 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.924756 w_k1_ADM_s003
                                 + 1.08594 w_k1_AXP_s003
                                 + 0.856127 w_k1_DIS_s003
                                 + 1.02737 w_k1_FDX_s003
                                 + 1.00806 w_k1_IBM_s003
                                 + 0.97299 w_k1_INTC_s003
                                 + 0.928829 w_k1_KO_s003
                                 + 0.977441 w_k1_MAT_s003
                                 + 0.978787 w_k1_MSFT_s003
                                 + 0.981937 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.702125 w_k1_ADM_s003
                                 + 1.56936 w_k1_AXP_s003 + 1.5912 w_k1_DIS_s003
                                 + 1.1109 w_k1_FDX_s003 + 1.82009 w_k1_IBM_s003
                                 + 1.19013 w_k1_INTC_s003 + 1.5248 w_k1_KO_s003
                                 + 2.14431 w_k1_MAT_s003
                                 + 1.26988 w_k1_MSFT_s003
                                 + 1.17604 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.918864 w_k1_ADM_s003
                                 + 1.617 w_k1_AXP_s003 + 0.830268 w_k1_DIS_s003
                                 + 1.27085 w_k1_FDX_s003
                                 + 0.939195 w_k1_IBM_s003
                                 + 1.01572 w_k1_INTC_s003
                                 + 0.80018 w_k1_KO_s003 + 1.24109 w_k1_MAT_s003
                                 + 0.876182 w_k1_MSFT_s003
                                 + 1.07484 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.629864 w_k1_ADM_s003
                                 + 1.28405 w_k1_AXP_s003
                                 + 1.22041 w_k1_DIS_s003
                                 + 1.06544 w_k1_FDX_s003
                                 + 1.35366 w_k1_IBM_s003
                                 + 1.18121 w_k1_INTC_s003
                                 + 1.12607 w_k1_KO_s003 + 1.25394 w_k1_MAT_s003
                                 + 1.15594 w_k1_MSFT_s003
                                 + 1.01829 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.48407 w_k1_ADM_s003
                                 + 1.92009 w_k1_AXP_s003
                                 + 1.73258 w_k1_DIS_s003
                                 + 1.62454 w_k1_FDX_s003
                                 + 1.13415 w_k1_IBM_s003
                                 + 1.54561 w_k1_INTC_s003
                                 + 1.23126 w_k1_KO_s003 + 1.12918 w_k1_MAT_s003
                                 + 1.54089 w_k1_MSFT_s003
                                 + 1.34033 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.559928 w_k1_ADM_s003
                                 + 0.708899 w_k1_AXP_s003
                                 + 0.645861 w_k1_DIS_s003
                                 + 0.803099 w_k1_FDX_s003
                                 + 0.726387 w_k1_IBM_s003
                                 + 0.848704 w_k1_INTC_s003
                                 + 0.631789 w_k1_KO_s003 + 0.6851 w_k1_MAT_s003
                                 + 0.742568 w_k1_MSFT_s003
                                 + 0.781289 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.698898 w_k1_ADM_s004
                                 + 0.56349 w_k1_AXP_s004
                                 + 0.613059 w_k1_DIS_s004
                                 + 0.787629 w_k1_FDX_s004
                                 + 0.683195 w_k1_IBM_s004
                                 + 0.990253 w_k1_INTC_s004
                                 + 0.711803 w_k1_KO_s004
                                 + 0.682471 w_k1_MAT_s004
                                 + 0.827113 w_k1_MSFT_s004
                                 + 0.90659 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.953499 w_k1_ADM_s004
                                 + 1.05956 w_k1_AXP_s004
                                 + 0.853623 w_k1_DIS_s004
                                 + 0.984733 w_k1_FDX_s004
                                 + 1.09127 w_k1_IBM_s004
                                 + 1.06492 w_k1_INTC_s004
                                 + 0.985871 w_k1_KO_s004 + 1.327 w_k1_MAT_s004
                                 + 1.06143 w_k1_MSFT_s004
                                 + 0.993442 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.55418 w_k1_ADM_s004
                                 + 1.14016 w_k1_AXP_s004
                                 + 1.18923 w_k1_DIS_s004
                                 + 1.21119 w_k1_FDX_s004
                                 + 0.755757 w_k1_IBM_s004
                                 + 1.27422 w_k1_INTC_s004 + 1.0999 w_k1_KO_s004
                                 + 0.804578 w_k1_MAT_s004
                                 + 1.2695 w_k1_MSFT_s004
                                 + 0.999025 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.09273 w_k1_ADM_s004
                                 + 0.870251 w_k1_AXP_s004
                                 + 1.01476 w_k1_DIS_s004 + 1.0364 w_k1_FDX_s004
                                 + 1.16653 w_k1_IBM_s004
                                 + 1.14523 w_k1_INTC_s004
                                 + 1.16554 w_k1_KO_s004
                                 + 0.990737 w_k1_MAT_s004
                                 + 1.08216 w_k1_MSFT_s004
                                 + 1.05533 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.46984 w_k1_ADM_s004
                                 + 1.51712 w_k1_AXP_s004
                                 + 1.60987 w_k1_DIS_s004
                                 + 1.34578 w_k1_FDX_s004
                                 + 1.09029 w_k1_IBM_s004
                                 + 1.34487 w_k1_INTC_s004
                                 + 1.25455 w_k1_KO_s004 + 1.31541 w_k1_MAT_s004
                                 + 1.38072 w_k1_MSFT_s004
                                 + 1.11815 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.46215 w_k1_ADM_s004
                                 + 1.35355 w_k1_AXP_s004
                                 + 1.23744 w_k1_DIS_s004
                                 + 1.15421 w_k1_FDX_s004
                                 + 1.44307 w_k1_IBM_s004
                                 + 1.54157 w_k1_INTC_s004
                                 + 1.13033 w_k1_KO_s004 + 1.65629 w_k1_MAT_s004
                                 + 1.17317 w_k1_MSFT_s004
                                 + 1.07075 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.913823 w_k1_ADM_s004
                                 + 0.782146 w_k1_AXP_s004
                                 + 0.78282 w_k1_DIS_s004
                                 + 0.945209 w_k1_FDX_s004
                                 + 0.874001 w_k1_IBM_s004
                                 + 1.07032 w_k1_INTC_s004
                                 + 0.967253 w_k1_KO_s004
                                 + 0.739941 w_k1_MAT_s004
                                 + 0.994929 w_k1_MSFT_s004
                                 + 0.955852 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.859205 w_k1_ADM_s004
                                 + 0.827081 w_k1_AXP_s004
                                 + 0.613576 w_k1_DIS_s004
                                 + 0.734875 w_k1_FDX_s004
                                 + 1.13079 w_k1_IBM_s004
                                 + 0.97719 w_k1_INTC_s004
                                 + 0.920215 w_k1_KO_s004
                                 + 0.871657 w_k1_MAT_s004
                                 + 0.88509 w_k1_MSFT_s004
                                 + 0.913205 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.52733 w_k1_ADM_s005
                                 + 1.41697 w_k1_AXP_s005
                                 + 1.66727 w_k1_DIS_s005
                                 + 1.41673 w_k1_FDX_s005
                                 + 1.10723 w_k1_IBM_s005
                                 + 1.38408 w_k1_INTC_s005
                                 + 1.33481 w_k1_KO_s005
                                 + 0.834117 w_k1_MAT_s005
                                 + 1.13131 w_k1_MSFT_s005
                                 + 1.21827 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.32491 w_k1_ADM_s005
                                 + 1.62979 w_k1_AXP_s005
                                 + 1.43279 w_k1_DIS_s005
                                 + 1.17047 w_k1_FDX_s005
                                 + 1.46025 w_k1_IBM_s005
                                 + 1.12793 w_k1_INTC_s005
                                 + 1.26042 w_k1_KO_s005 + 1.52414 w_k1_MAT_s005
                                 + 0.984085 w_k1_MSFT_s005
                                 + 1.12097 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.966377 w_k1_ADM_s005
                                 + 0.775968 w_k1_AXP_s005
                                 + 1.10063 w_k1_DIS_s005
                                 + 0.999152 w_k1_FDX_s005
                                 + 0.731038 w_k1_IBM_s005
                                 + 1.13096 w_k1_INTC_s005
                                 + 0.895143 w_k1_KO_s005
                                 + 0.89493 w_k1_MAT_s005
                                 + 1.06373 w_k1_MSFT_s005
                                 + 1.01517 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.02232 w_k1_ADM_s005
                                 + 1.09095 w_k1_AXP_s005
                                 + 1.05557 w_k1_DIS_s005 + 1.0582 w_k1_FDX_s005
                                 + 1.27099 w_k1_IBM_s005
                                 + 1.02559 w_k1_INTC_s005 + 1.0036 w_k1_KO_s005
                                 + 0.936191 w_k1_MAT_s005
                                 + 0.844513 w_k1_MSFT_s005
                                 + 1.12403 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.03713 w_k1_ADM_s005
                                 + 1.0193 w_k1_AXP_s005
                                 + 0.931259 w_k1_DIS_s005
                                 + 0.87593 w_k1_FDX_s005
                                 + 1.20492 w_k1_IBM_s005
                                 + 1.04146 w_k1_INTC_s005
                                 + 0.972823 w_k1_KO_s005
                                 + 1.09808 w_k1_MAT_s005
                                 + 0.810297 w_k1_MSFT_s005
                                 + 1.12113 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.01903 w_k1_ADM_s005
                                 + 0.863294 w_k1_AXP_s005
                                 + 0.823587 w_k1_DIS_s005
                                 + 0.8949 w_k1_FDX_s005
                                 + 0.697165 w_k1_IBM_s005
                                 + 1.08856 w_k1_INTC_s005
                                 + 0.689455 w_k1_KO_s005
                                 + 0.529977 w_k1_MAT_s005
                                 + 0.808159 w_k1_MSFT_s005
                                 + 0.92787 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.8724 w_k1_ADM_s005
                                 + 0.735367 w_k1_AXP_s005
                                 + 0.660807 w_k1_DIS_s005
                                 + 0.743186 w_k1_FDX_s005
                                 + 0.908554 w_k1_IBM_s005
                                 + 1.00069 w_k1_INTC_s005
                                 + 0.829195 w_k1_KO_s005
                                 + 0.789443 w_k1_MAT_s005
                                 + 0.706918 w_k1_MSFT_s005
                                 + 0.986269 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.956828 w_k1_ADM_s005
                                 + 0.935204 w_k1_AXP_s005
                                 + 1.3626 w_k1_DIS_s005 + 1.11608 w_k1_FDX_s005
                                 + 1.03731 w_k1_IBM_s005
                                 + 1.23736 w_k1_INTC_s005
                                 + 1.16044 w_k1_KO_s005
                                 + 0.812806 w_k1_MAT_s005
                                 + 1.0676 w_k1_MSFT_s005 + 1.05856 w_k1_PG_s005
                                 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.34356 w_k1_ADM_s006
                                 + 1.47609 w_k1_AXP_s006
                                 + 1.49617 w_k1_DIS_s006
                                 + 1.52901 w_k1_FDX_s006
                                 + 0.989869 w_k1_IBM_s006
                                 + 1.55115 w_k1_INTC_s006
                                 + 1.27649 w_k1_KO_s006 + 1.07545 w_k1_MAT_s006
                                 + 1.47159 w_k1_MSFT_s006
                                 + 1.24596 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.959843 w_k1_ADM_s006
                                 + 1.12834 w_k1_AXP_s006
                                 + 1.65842 w_k1_DIS_s006
                                 + 1.16066 w_k1_FDX_s006
                                 + 1.53534 w_k1_IBM_s006
                                 + 1.55998 w_k1_INTC_s006
                                 + 1.67097 w_k1_KO_s006 + 1.25416 w_k1_MAT_s006
                                 + 1.46981 w_k1_MSFT_s006
                                 + 1.34039 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.18755 w_k1_ADM_s006
                                 + 1.07194 w_k1_AXP_s006 + 1.0406 w_k1_DIS_s006
                                 + 1.11764 w_k1_FDX_s006
                                 + 0.850692 w_k1_IBM_s006
                                 + 1.16681 w_k1_INTC_s006
                                 + 0.931304 w_k1_KO_s006
                                 + 0.626604 w_k1_MAT_s006
                                 + 1.15111 w_k1_MSFT_s006
                                 + 0.99706 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.870814 w_k1_ADM_s006
                                 + 0.749913 w_k1_AXP_s006
                                 + 0.72314 w_k1_DIS_s006
                                 + 0.798366 w_k1_FDX_s006
                                 + 0.976266 w_k1_IBM_s006
                                 + 0.877312 w_k1_INTC_s006
                                 + 0.938584 w_k1_KO_s006
                                 + 0.698068 w_k1_MAT_s006
                                 + 0.912863 w_k1_MSFT_s006
                                 + 0.968314 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.959262 w_k1_ADM_s006
                                 + 1.29378 w_k1_AXP_s006
                                 + 1.43937 w_k1_DIS_s006
                                 + 1.21158 w_k1_FDX_s006
                                 + 1.28501 w_k1_IBM_s006
                                 + 1.56903 w_k1_INTC_s006
                                 + 1.40336 w_k1_KO_s006 + 1.13917 w_k1_MAT_s006
                                 + 1.43574 w_k1_MSFT_s006
                                 + 1.14616 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.853403 w_k1_ADM_s006
                                 + 0.906109 w_k1_AXP_s006
                                 + 1.00682 w_k1_DIS_s006
                                 + 0.940491 w_k1_FDX_s006 + 1.225 w_k1_IBM_s006
                                 + 1.28 w_k1_INTC_s006 + 1.27532 w_k1_KO_s006
                                 + 0.877696 w_k1_MAT_s006
                                 + 1.11796 w_k1_MSFT_s006
                                 + 1.13152 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.790823 w_k1_ADM_s006
                                 + 0.511519 w_k1_AXP_s006
                                 + 0.610455 w_k1_DIS_s006
                                 + 0.722563 w_k1_FDX_s006
                                 + 0.673695 w_k1_IBM_s006
                                 + 0.939433 w_k1_INTC_s006
                                 + 0.779712 w_k1_KO_s006
                                 + 0.404651 w_k1_MAT_s006
                                 + 0.769862 w_k1_MSFT_s006
                                 + 0.872958 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.101 w_k1_ADM_s006
                                 + 0.763474 w_k1_AXP_s006
                                 + 0.894147 w_k1_DIS_s006
                                 + 0.940146 w_k1_FDX_s006
                                 + 0.792006 w_k1_IBM_s006
                                 + 0.933436 w_k1_INTC_s006
                                 + 1.0717 w_k1_KO_s006 + 0.68302 w_k1_MAT_s006
                                 + 0.985675 w_k1_MSFT_s006
                                 + 0.971161 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.932477 w_k1_ADM_s007
                                 + 1.07848 w_k1_AXP_s007
                                 + 1.03656 w_k1_DIS_s007
                                 + 1.13174 w_k1_FDX_s007
                                 + 1.01639 w_k1_IBM_s007
                                 + 0.926905 w_k1_INTC_s007
                                 + 1.01687 w_k1_KO_s007 + 1.8765 w_k1_MAT_s007
                                 + 1.07061 w_k1_MSFT_s007
                                 + 1.04637 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.11851 w_k1_ADM_s007
                                 + 1.01253 w_k1_AXP_s007
                                 + 1.06794 w_k1_DIS_s007
                                 + 1.13778 w_k1_FDX_s007
                                 + 0.843701 w_k1_IBM_s007
                                 + 0.847811 w_k1_INTC_s007
                                 + 0.960537 w_k1_KO_s007
                                 + 1.19015 w_k1_MAT_s007
                                 + 1.09516 w_k1_MSFT_s007
                                 + 1.13277 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.09255 w_k1_ADM_s007
                                 + 1.04117 w_k1_AXP_s007
                                 + 0.934716 w_k1_DIS_s007
                                 + 0.86325 w_k1_FDX_s007
                                 + 1.17624 w_k1_IBM_s007
                                 + 0.621435 w_k1_INTC_s007
                                 + 1.09819 w_k1_KO_s007 + 2.21591 w_k1_MAT_s007
                                 + 0.926684 w_k1_MSFT_s007
                                 + 0.906432 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.762226 w_k1_ADM_s007
                                 + 0.73438 w_k1_AXP_s007
                                 + 0.755552 w_k1_DIS_s007
                                 + 0.76684 w_k1_FDX_s007
                                 + 0.721613 w_k1_IBM_s007
                                 + 0.902903 w_k1_INTC_s007
                                 + 0.856784 w_k1_KO_s007
                                 + 0.786565 w_k1_MAT_s007
                                 + 0.839429 w_k1_MSFT_s007
                                 + 0.889885 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.886173 w_k1_ADM_s007
                                 + 0.705266 w_k1_AXP_s007
                                 + 0.988404 w_k1_DIS_s007
                                 + 0.963813 w_k1_FDX_s007
                                 + 0.628536 w_k1_IBM_s007
                                 + 0.958211 w_k1_INTC_s007
                                 + 0.881562 w_k1_KO_s007
                                 + 0.602283 w_k1_MAT_s007
                                 + 0.937089 w_k1_MSFT_s007
                                 + 0.832414 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.87024 w_k1_ADM_s007
                                 + 0.699041 w_k1_AXP_s007
                                 + 0.712425 w_k1_DIS_s007
                                 + 0.793877 w_k1_FDX_s007
                                 + 0.677984 w_k1_IBM_s007
                                 + 0.745889 w_k1_INTC_s007
                                 + 0.795983 w_k1_KO_s007
                                 + 0.691784 w_k1_MAT_s007
                                 + 0.788455 w_k1_MSFT_s007
                                 + 0.845231 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.782977 w_k1_ADM_s007
                                 + 0.811378 w_k1_AXP_s007
                                 + 0.955251 w_k1_DIS_s007
                                 + 0.872849 w_k1_FDX_s007
                                 + 1.14968 w_k1_IBM_s007
                                 + 0.965695 w_k1_INTC_s007
                                 + 1.12583 w_k1_KO_s007 + 1.36925 w_k1_MAT_s007
                                 + 0.872838 w_k1_MSFT_s007
                                 + 0.971079 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.24447 w_k1_ADM_s007
                                 + 1.42474 w_k1_AXP_s007 + 1.4871 w_k1_DIS_s007
                                 + 1.17289 w_k1_FDX_s007
                                 + 1.27126 w_k1_IBM_s007
                                 + 0.93394 w_k1_INTC_s007 + 1.4091 w_k1_KO_s007
                                 + 3.00782 w_k1_MAT_s007
                                 + 1.19609 w_k1_MSFT_s007
                                 + 1.22261 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.29915 w_k1_ADM_s010
                                 + 1.24494 w_k1_AXP_s010
                                 + 1.53595 w_k1_DIS_s010
                                 + 1.51844 w_k1_FDX_s010
                                 + 1.11744 w_k1_IBM_s010
                                 + 1.25066 w_k1_INTC_s010
                                 + 1.10593 w_k1_KO_s010
                                 + 0.835447 w_k1_MAT_s010
                                 + 1.38988 w_k1_MSFT_s010
                                 + 1.00542 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.14766 w_k1_ADM_s010
                                 + 1.18625 w_k1_AXP_s010
                                 + 1.22151 w_k1_DIS_s010 + 1.1747 w_k1_FDX_s010
                                 + 1.2585 w_k1_IBM_s010
                                 + 1.07871 w_k1_INTC_s010
                                 + 1.16859 w_k1_KO_s010 + 1.06929 w_k1_MAT_s010
                                 + 1.21294 w_k1_MSFT_s010
                                 + 1.11171 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.10901 w_k1_ADM_s010
                                 + 0.697175 w_k1_AXP_s010
                                 + 1.05304 w_k1_DIS_s010
                                 + 0.976239 w_k1_FDX_s010
                                 + 0.937256 w_k1_IBM_s010
                                 + 0.902385 w_k1_INTC_s010
                                 + 0.832197 w_k1_KO_s010
                                 + 0.600476 w_k1_MAT_s010
                                 + 1.05861 w_k1_MSFT_s010
                                 + 0.909554 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.16798 w_k1_ADM_s010
                                 + 1.92014 w_k1_AXP_s010
                                 + 1.90483 w_k1_DIS_s010
                                 + 1.62217 w_k1_FDX_s010
                                 + 1.66092 w_k1_IBM_s010
                                 + 1.28285 w_k1_INTC_s010
                                 + 1.38457 w_k1_KO_s010 + 1.9031 w_k1_MAT_s010
                                 + 1.58806 w_k1_MSFT_s010
                                 + 1.19796 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.867624 w_k1_ADM_s010
                                 + 0.805827 w_k1_AXP_s010
                                 + 0.783352 w_k1_DIS_s010
                                 + 0.849358 w_k1_FDX_s010
                                 + 0.94112 w_k1_IBM_s010
                                 + 0.882724 w_k1_INTC_s010
                                 + 0.73175 w_k1_KO_s010
                                 + 0.681805 w_k1_MAT_s010
                                 + 0.912209 w_k1_MSFT_s010
                                 + 0.791566 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.823926 w_k1_ADM_s010
                                 + 0.591622 w_k1_AXP_s010
                                 + 0.542848 w_k1_DIS_s010
                                 + 0.665766 w_k1_FDX_s010
                                 + 0.601375 w_k1_IBM_s010
                                 + 0.597018 w_k1_INTC_s010
                                 + 0.623152 w_k1_KO_s010
                                 + 0.377143 w_k1_MAT_s010
                                 + 0.679508 w_k1_MSFT_s010
                                 + 0.674725 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.87353 w_k1_ADM_s010
                                 + 0.771377 w_k1_AXP_s010
                                 + 0.61681 w_k1_DIS_s010
                                 + 0.683758 w_k1_FDX_s010
                                 + 1.25145 w_k1_IBM_s010
                                 + 0.852141 w_k1_INTC_s010
                                 + 0.785514 w_k1_KO_s010
                                 + 0.768053 w_k1_MAT_s010
                                 + 0.844915 w_k1_MSFT_s010
                                 + 0.754116 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.15831 w_k1_ADM_s010
                                 + 1.09549 w_k1_AXP_s010
                                 + 0.803367 w_k1_DIS_s010
                                 + 1.05431 w_k1_FDX_s010
                                 + 0.930668 w_k1_IBM_s010
                                 + 0.835169 w_k1_INTC_s010
                                 + 0.823104 w_k1_KO_s010
                                 + 0.725615 w_k1_MAT_s010
                                 + 1.0196 w_k1_MSFT_s010
                                 + 0.927739 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.794325 w_k1_ADM_s011
                                 + 1.14007 w_k1_AXP_s011
                                 + 0.759545 w_k1_DIS_s011
                                 + 0.978536 w_k1_FDX_s011
                                 + 0.967482 w_k1_IBM_s011
                                 + 0.85694 w_k1_INTC_s011
                                 + 0.851538 w_k1_KO_s011
                                 + 1.00307 w_k1_MAT_s011
                                 + 0.830577 w_k1_MSFT_s011
                                 + 0.991885 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.26163 w_k1_ADM_s011
                                 + 1.66014 w_k1_AXP_s011
                                 + 1.36017 w_k1_DIS_s011
                                 + 1.43396 w_k1_FDX_s011
                                 + 1.02854 w_k1_IBM_s011
                                 + 1.01572 w_k1_INTC_s011
                                 + 1.24023 w_k1_KO_s011 + 1.55934 w_k1_MAT_s011
                                 + 1.14793 w_k1_MSFT_s011
                                 + 1.11821 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.48217 w_k1_ADM_s011
                                 + 1.51069 w_k1_AXP_s011
                                 + 1.63422 w_k1_DIS_s011
                                 + 1.24918 w_k1_FDX_s011
                                 + 1.81481 w_k1_IBM_s011
                                 + 1.01731 w_k1_INTC_s011
                                 + 1.70487 w_k1_KO_s011 + 1.92655 w_k1_MAT_s011
                                 + 1.29644 w_k1_MSFT_s011
                                 + 1.29965 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.812412 w_k1_ADM_s011
                                 + 0.739226 w_k1_AXP_s011
                                 + 0.784659 w_k1_DIS_s011
                                 + 0.862533 w_k1_FDX_s011
                                 + 0.645672 w_k1_IBM_s011
                                 + 0.777813 w_k1_INTC_s011
                                 + 0.765102 w_k1_KO_s011
                                 + 0.513662 w_k1_MAT_s011
                                 + 0.834371 w_k1_MSFT_s011
                                 + 0.870668 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.931046 w_k1_ADM_s011
                                 + 1.15473 w_k1_AXP_s011
                                 + 1.14915 w_k1_DIS_s011
                                 + 0.920401 w_k1_FDX_s011
                                 + 1.5439 w_k1_IBM_s011
                                 + 0.727973 w_k1_INTC_s011
                                 + 1.20575 w_k1_KO_s011 + 1.59604 w_k1_MAT_s011
                                 + 0.939758 w_k1_MSFT_s011
                                 + 1.06819 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.724414 w_k1_ADM_s011
                                 + 0.604849 w_k1_AXP_s011
                                 + 0.745969 w_k1_DIS_s011
                                 + 0.751855 w_k1_FDX_s011
                                 + 0.991205 w_k1_IBM_s011
                                 + 0.741293 w_k1_INTC_s011
                                 + 0.90074 w_k1_KO_s011 + 1.03387 w_k1_MAT_s011
                                 + 0.813894 w_k1_MSFT_s011
                                 + 0.913786 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.21717 w_k1_ADM_s011
                                 + 1.19347 w_k1_AXP_s011
                                 + 1.25851 w_k1_DIS_s011 + 1.1671 w_k1_FDX_s011
                                 + 0.83566 w_k1_IBM_s011
                                 + 0.8659 w_k1_INTC_s011
                                 + 0.983458 w_k1_KO_s011
                                 + 1.09118 w_k1_MAT_s011
                                 + 1.08317 w_k1_MSFT_s011
                                 + 1.16672 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.11935 w_k1_ADM_s011
                                 + 1.29355 w_k1_AXP_s011
                                 + 1.29841 w_k1_DIS_s011
                                 + 1.13924 w_k1_FDX_s011 + 1.1919 w_k1_IBM_s011
                                 + 0.883795 w_k1_INTC_s011
                                 + 1.21444 w_k1_KO_s011 + 1.41986 w_k1_MAT_s011
                                 + 1.04189 w_k1_MSFT_s011 + 1.1282 w_k1_PG_s011
                                 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.804447 w_k1_ADM_s012
                                 + 0.818353 w_k1_AXP_s012
                                 + 0.843075 w_k1_DIS_s012
                                 + 0.779742 w_k1_FDX_s012
                                 + 0.976834 w_k1_IBM_s012
                                 + 0.876848 w_k1_INTC_s012
                                 + 0.699511 w_k1_KO_s012
                                 + 0.969355 w_k1_MAT_s012
                                 + 0.783414 w_k1_MSFT_s012
                                 + 0.836787 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.807282 w_k1_ADM_s012
                                 + 0.824212 w_k1_AXP_s012
                                 + 0.757343 w_k1_DIS_s012
                                 + 0.901639 w_k1_FDX_s012
                                 + 0.881588 w_k1_IBM_s012
                                 + 1.02287 w_k1_INTC_s012
                                 + 0.80743 w_k1_KO_s012
                                 + 0.673561 w_k1_MAT_s012
                                 + 0.97863 w_k1_MSFT_s012
                                 + 0.865857 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.07794 w_k1_ADM_s012
                                 + 0.902384 w_k1_AXP_s012
                                 + 0.983362 w_k1_DIS_s012
                                 + 1.02754 w_k1_FDX_s012
                                 + 0.860595 w_k1_IBM_s012
                                 + 0.936403 w_k1_INTC_s012
                                 + 0.742562 w_k1_KO_s012
                                 + 0.891644 w_k1_MAT_s012
                                 + 0.884757 w_k1_MSFT_s012
                                 + 0.942206 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.03209 w_k1_ADM_s012
                                 + 1.12296 w_k1_AXP_s012
                                 + 1.14231 w_k1_DIS_s012
                                 + 1.17436 w_k1_FDX_s012
                                 + 1.02268 w_k1_IBM_s012
                                 + 1.0962 w_k1_INTC_s012 + 1.06635 w_k1_KO_s012
                                 + 1.30095 w_k1_MAT_s012
                                 + 1.08922 w_k1_MSFT_s012
                                 + 1.04877 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.09813 w_k1_ADM_s012
                                 + 1.24707 w_k1_AXP_s012
                                 + 1.70066 w_k1_DIS_s012
                                 + 1.44358 w_k1_FDX_s012
                                 + 1.14156 w_k1_IBM_s012
                                 + 1.23085 w_k1_INTC_s012
                                 + 1.05804 w_k1_KO_s012 + 1.59895 w_k1_MAT_s012
                                 + 1.4299 w_k1_MSFT_s012 + 1.23538 w_k1_PG_s012
                                 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.787416 w_k1_ADM_s012
                                 + 1.20524 w_k1_AXP_s012
                                 + 1.53394 w_k1_DIS_s012
                                 + 1.10622 w_k1_FDX_s012
                                 + 1.66844 w_k1_IBM_s012
                                 + 0.938005 w_k1_INTC_s012
                                 + 1.49733 w_k1_KO_s012 + 2.30755 w_k1_MAT_s012
                                 + 1.02796 w_k1_MSFT_s012
                                 + 1.15729 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.755902 w_k1_ADM_s012
                                 + 0.622283 w_k1_AXP_s012
                                 + 0.702233 w_k1_DIS_s012
                                 + 0.666806 w_k1_FDX_s012
                                 + 0.914045 w_k1_IBM_s012
                                 + 0.822573 w_k1_INTC_s012
                                 + 0.717023 w_k1_KO_s012
                                 + 0.533006 w_k1_MAT_s012
                                 + 0.751439 w_k1_MSFT_s012
                                 + 0.723921 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.13251 w_k1_ADM_s012
                                 + 1.05621 w_k1_AXP_s012
                                 + 1.01674 w_k1_DIS_s012
                                 + 0.979747 w_k1_FDX_s012
                                 + 1.34086 w_k1_IBM_s012
                                 + 0.83179 w_k1_INTC_s012
                                 + 1.07275 w_k1_KO_s012 + 1.71309 w_k1_MAT_s012
                                 + 0.889273 w_k1_MSFT_s012
                                 + 0.94182 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.19457 w_k1_ADM_s013
                                 + 1.0248 w_k1_AXP_s013
                                 + 0.778623 w_k1_DIS_s013
                                 + 0.93052 w_k1_FDX_s013
                                 + 1.07079 w_k1_IBM_s013
                                 + 0.797729 w_k1_INTC_s013
                                 + 0.87017 w_k1_KO_s013 + 1.09671 w_k1_MAT_s013
                                 + 0.764075 w_k1_MSFT_s013
                                 + 1.09783 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.03989 w_k1_ADM_s013
                                 + 1.47085 w_k1_AXP_s013
                                 + 1.48269 w_k1_DIS_s013
                                 + 1.38274 w_k1_FDX_s013
                                 + 1.11761 w_k1_IBM_s013
                                 + 1.32656 w_k1_INTC_s013
                                 + 1.07235 w_k1_KO_s013 + 1.14693 w_k1_MAT_s013
                                 + 1.25108 w_k1_MSFT_s013
                                 + 1.25785 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.20682 w_k1_ADM_s013
                                 + 1.13096 w_k1_AXP_s013
                                 + 0.736522 w_k1_DIS_s013
                                 + 0.9796 w_k1_FDX_s013
                                 + 0.534185 w_k1_IBM_s013
                                 + 0.848829 w_k1_INTC_s013
                                 + 0.683668 w_k1_KO_s013
                                 + 0.704638 w_k1_MAT_s013
                                 + 0.854172 w_k1_MSFT_s013
                                 + 0.895937 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.02798 w_k1_ADM_s013
                                 + 1.12996 w_k1_AXP_s013
                                 + 0.949712 w_k1_DIS_s013
                                 + 1.04317 w_k1_FDX_s013
                                 + 0.908653 w_k1_IBM_s013
                                 + 1.11894 w_k1_INTC_s013
                                 + 0.962401 w_k1_KO_s013
                                 + 0.954075 w_k1_MAT_s013
                                 + 1.02794 w_k1_MSFT_s013
                                 + 1.00578 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.1126 w_k1_ADM_s013
                                 + 0.779642 w_k1_AXP_s013
                                 + 0.679524 w_k1_DIS_s013
                                 + 0.794212 w_k1_FDX_s013
                                 + 0.841149 w_k1_IBM_s013
                                 + 0.854262 w_k1_INTC_s013
                                 + 0.845133 w_k1_KO_s013
                                 + 0.667666 w_k1_MAT_s013
                                 + 0.800174 w_k1_MSFT_s013
                                 + 0.926375 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.20064 w_k1_ADM_s013
                                 + 1.21709 w_k1_AXP_s013
                                 + 1.25941 w_k1_DIS_s013
                                 + 1.08598 w_k1_FDX_s013
                                 + 1.73124 w_k1_IBM_s013
                                 + 0.872043 w_k1_INTC_s013
                                 + 1.18041 w_k1_KO_s013 + 1.29866 w_k1_MAT_s013
                                 + 1.01013 w_k1_MSFT_s013
                                 + 1.07799 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.892195 w_k1_ADM_s013
                                 + 1.14437 w_k1_AXP_s013
                                 + 1.23021 w_k1_DIS_s013
                                 + 1.01157 w_k1_FDX_s013
                                 + 1.19439 w_k1_IBM_s013
                                 + 0.821649 w_k1_INTC_s013
                                 + 1.10788 w_k1_KO_s013 + 1.21041 w_k1_MAT_s013
                                 + 0.964683 w_k1_MSFT_s013
                                 + 0.984877 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.00018 w_k1_ADM_s013
                                 + 1.00689 w_k1_AXP_s013
                                 + 0.824372 w_k1_DIS_s013
                                 + 0.808557 w_k1_FDX_s013
                                 + 1.43362 w_k1_IBM_s013
                                 + 0.849303 w_k1_INTC_s013
                                 + 1.05741 w_k1_KO_s013 + 1.29085 w_k1_MAT_s013
                                 + 0.885492 w_k1_MSFT_s013
                                 + 1.04567 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.19946 w_k1_ADM_s014
                                 + 1.46064 w_k1_AXP_s014 + 1.18 w_k1_DIS_s014
                                 + 1.22266 w_k1_FDX_s014
                                 + 1.10177 w_k1_IBM_s014
                                 + 1.15872 w_k1_INTC_s014
                                 + 1.02457 w_k1_KO_s014 + 1.18339 w_k1_MAT_s014
                                 + 1.00084 w_k1_MSFT_s014
                                 + 0.993551 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.839977 w_k1_ADM_s014
                                 + 1.0728 w_k1_AXP_s014
                                 + 0.920254 w_k1_DIS_s014
                                 + 1.05028 w_k1_FDX_s014
                                 + 0.72363 w_k1_IBM_s014
                                 + 1.0172 w_k1_INTC_s014 + 0.75057 w_k1_KO_s014
                                 + 0.792143 w_k1_MAT_s014
                                 + 1.0267 w_k1_MSFT_s014
                                 + 0.918546 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.03216 w_k1_ADM_s014
                                 + 1.19166 w_k1_AXP_s014
                                 + 1.11755 w_k1_DIS_s014
                                 + 1.07461 w_k1_FDX_s014
                                 + 0.957371 w_k1_IBM_s014
                                 + 1.09938 w_k1_INTC_s014
                                 + 0.990152 w_k1_KO_s014
                                 + 0.876914 w_k1_MAT_s014
                                 + 0.880603 w_k1_MSFT_s014
                                 + 1.00654 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.732286 w_k1_ADM_s014
                                 + 0.796193 w_k1_AXP_s014
                                 + 0.907405 w_k1_DIS_s014
                                 + 0.878929 w_k1_FDX_s014
                                 + 0.998839 w_k1_IBM_s014
                                 + 0.929756 w_k1_INTC_s014
                                 + 0.826695 w_k1_KO_s014
                                 + 0.789057 w_k1_MAT_s014
                                 + 0.898706 w_k1_MSFT_s014
                                 + 0.919072 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.05436 w_k1_ADM_s014
                                 + 1.43312 w_k1_AXP_s014
                                 + 1.43077 w_k1_DIS_s014
                                 + 1.40557 w_k1_FDX_s014
                                 + 0.851587 w_k1_IBM_s014
                                 + 1.28947 w_k1_INTC_s014
                                 + 0.976095 w_k1_KO_s014
                                 + 0.82692 w_k1_MAT_s014
                                 + 1.22012 w_k1_MSFT_s014
                                 + 1.05165 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.650007 w_k1_ADM_s014
                                 + 0.656384 w_k1_AXP_s014
                                 + 0.615875 w_k1_DIS_s014
                                 + 0.785861 w_k1_FDX_s014
                                 + 0.636851 w_k1_IBM_s014
                                 + 0.73318 w_k1_INTC_s014
                                 + 0.608831 w_k1_KO_s014
                                 + 0.681629 w_k1_MAT_s014
                                 + 0.784502 w_k1_MSFT_s014
                                 + 0.860362 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.57161 w_k1_ADM_s014
                                 + 0.896055 w_k1_AXP_s014
                                 + 0.902098 w_k1_DIS_s014
                                 + 0.873262 w_k1_FDX_s014
                                 + 1.34415 w_k1_IBM_s014
                                 + 1.13829 w_k1_INTC_s014
                                 + 0.992455 w_k1_KO_s014
                                 + 1.04053 w_k1_MAT_s014
                                 + 0.888426 w_k1_MSFT_s014
                                 + 1.04397 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.1149 w_k1_ADM_s014
                                 + 1.3769 w_k1_AXP_s014 + 1.52236 w_k1_DIS_s014
                                 + 1.16955 w_k1_FDX_s014
                                 + 1.67737 w_k1_IBM_s014
                                 + 1.09794 w_k1_INTC_s014
                                 + 1.40012 w_k1_KO_s014 + 1.39249 w_k1_MAT_s014
                                 + 1.0648 w_k1_MSFT_s014 + 1.11923 w_k1_PG_s014
                                 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.10958 w_k1_ADM_s015
                                 + 0.79254 w_k1_AXP_s015
                                 + 0.835109 w_k1_DIS_s015
                                 + 0.895114 w_k1_FDX_s015
                                 + 1.14016 w_k1_IBM_s015 + 0.927 w_k1_INTC_s015
                                 + 1.10288 w_k1_KO_s015
                                 + 0.914514 w_k1_MAT_s015
                                 + 0.876939 w_k1_MSFT_s015
                                 + 0.935247 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.46793 w_k1_ADM_s015
                                 + 1.33375 w_k1_AXP_s015
                                 + 1.26923 w_k1_DIS_s015
                                 + 1.06712 w_k1_FDX_s015 + 1.4324 w_k1_IBM_s015
                                 + 0.937758 w_k1_INTC_s015
                                 + 1.28836 w_k1_KO_s015 + 1.68529 w_k1_MAT_s015
                                 + 1.11325 w_k1_MSFT_s015
                                 + 1.21133 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.10939 w_k1_ADM_s015
                                 + 1.52729 w_k1_AXP_s015
                                 + 0.825148 w_k1_DIS_s015
                                 + 0.905718 w_k1_FDX_s015
                                 + 1.29337 w_k1_IBM_s015
                                 + 0.909349 w_k1_INTC_s015
                                 + 0.892943 w_k1_KO_s015
                                 + 1.93556 w_k1_MAT_s015
                                 + 0.792546 w_k1_MSFT_s015
                                 + 0.92353 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.980955 w_k1_ADM_s015
                                 + 1.17065 w_k1_AXP_s015
                                 + 0.823673 w_k1_DIS_s015
                                 + 1.04519 w_k1_FDX_s015
                                 + 0.712172 w_k1_IBM_s015
                                 + 1.04089 w_k1_INTC_s015
                                 + 0.795103 w_k1_KO_s015
                                 + 0.919129 w_k1_MAT_s015
                                 + 1.09827 w_k1_MSFT_s015
                                 + 0.911717 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.83333 w_k1_ADM_s015
                                 + 0.851501 w_k1_AXP_s015
                                 + 0.842044 w_k1_DIS_s015
                                 + 0.887055 w_k1_FDX_s015
                                 + 0.641329 w_k1_IBM_s015
                                 + 0.847861 w_k1_INTC_s015
                                 + 0.720899 w_k1_KO_s015
                                 + 0.738232 w_k1_MAT_s015
                                 + 0.843953 w_k1_MSFT_s015
                                 + 0.831607 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.987952 w_k1_ADM_s015
                                 + 1.08724 w_k1_AXP_s015
                                 + 1.42897 w_k1_DIS_s015
                                 + 1.16242 w_k1_FDX_s015
                                 + 0.862284 w_k1_IBM_s015
                                 + 1.24359 w_k1_INTC_s015
                                 + 1.24752 w_k1_KO_s015
                                 + 0.991819 w_k1_MAT_s015
                                 + 1.47603 w_k1_MSFT_s015
                                 + 1.07232 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.757669 w_k1_ADM_s015
                                 + 1.0723 w_k1_AXP_s015
                                 + 0.970219 w_k1_DIS_s015
                                 + 0.936583 w_k1_FDX_s015
                                 + 0.912502 w_k1_IBM_s015
                                 + 0.98506 w_k1_INTC_s015
                                 + 0.92355 w_k1_KO_s015 + 1.53447 w_k1_MAT_s015
                                 + 0.884869 w_k1_MSFT_s015
                                 + 0.922886 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.12328 w_k1_ADM_s015
                                 + 1.80046 w_k1_AXP_s015
                                 + 1.91173 w_k1_DIS_s015
                                 + 1.24464 w_k1_FDX_s015
                                 + 1.65158 w_k1_IBM_s015
                                 + 1.35573 w_k1_INTC_s015
                                 + 1.48087 w_k1_KO_s015 + 2.19457 w_k1_MAT_s015
                                 + 1.36556 w_k1_MSFT_s015
                                 + 1.19775 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.287 w_k1_ADM_s016
                                 + 1.48745 w_k1_AXP_s016
                                 + 1.51676 w_k1_DIS_s016
                                 + 1.15639 w_k1_FDX_s016
                                 + 1.53652 w_k1_IBM_s016
                                 + 1.15883 w_k1_INTC_s016
                                 + 1.37793 w_k1_KO_s016 + 1.67875 w_k1_MAT_s016
                                 + 1.29455 w_k1_MSFT_s016
                                 + 1.14288 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.4984 w_k1_ADM_s016
                                 + 1.43804 w_k1_AXP_s016
                                 + 1.77498 w_k1_DIS_s016
                                 + 1.40854 w_k1_FDX_s016
                                 + 1.11708 w_k1_IBM_s016
                                 + 1.32776 w_k1_INTC_s016
                                 + 1.18365 w_k1_KO_s016 + 0.67957 w_k1_MAT_s016
                                 + 1.34606 w_k1_MSFT_s016
                                 + 1.19542 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.946552 w_k1_ADM_s016
                                 + 0.898371 w_k1_AXP_s016
                                 + 0.881194 w_k1_DIS_s016
                                 + 0.864532 w_k1_FDX_s016
                                 + 1.13824 w_k1_IBM_s016
                                 + 0.96457 w_k1_INTC_s016
                                 + 0.968527 w_k1_KO_s016
                                 + 0.928539 w_k1_MAT_s016
                                 + 0.89838 w_k1_MSFT_s016
                                 + 1.01635 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.884291 w_k1_ADM_s016
                                 + 0.846734 w_k1_AXP_s016
                                 + 0.669559 w_k1_DIS_s016
                                 + 0.784899 w_k1_FDX_s016
                                 + 0.774784 w_k1_IBM_s016
                                 + 0.745403 w_k1_INTC_s016
                                 + 0.628879 w_k1_KO_s016
                                 + 0.57396 w_k1_MAT_s016
                                 + 0.79696 w_k1_MSFT_s016
                                 + 0.827024 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.04507 w_k1_ADM_s016
                                 + 1.04373 w_k1_AXP_s016 + 1.2589 w_k1_DIS_s016
                                 + 1.09998 w_k1_FDX_s016
                                 + 0.958516 w_k1_IBM_s016
                                 + 1.17219 w_k1_INTC_s016
                                 + 0.943461 w_k1_KO_s016
                                 + 0.52674 w_k1_MAT_s016
                                 + 1.08153 w_k1_MSFT_s016
                                 + 0.951956 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.23612 w_k1_ADM_s016
                                 + 1.18084 w_k1_AXP_s016
                                 + 0.98599 w_k1_DIS_s016
                                 + 1.06848 w_k1_FDX_s016
                                 + 0.945128 w_k1_IBM_s016
                                 + 0.99788 w_k1_INTC_s016
                                 + 0.956919 w_k1_KO_s016
                                 + 0.848695 w_k1_MAT_s016
                                 + 0.964042 w_k1_MSFT_s016
                                 + 0.993769 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.909415 w_k1_ADM_s016
                                 + 0.758679 w_k1_AXP_s016
                                 + 0.879142 w_k1_DIS_s016
                                 + 0.995595 w_k1_FDX_s016
                                 + 0.749514 w_k1_IBM_s016
                                 + 1.15465 w_k1_INTC_s016
                                 + 0.909201 w_k1_KO_s016
                                 + 0.620255 w_k1_MAT_s016
                                 + 1.08277 w_k1_MSFT_s016
                                 + 0.99683 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.27093 w_k1_ADM_s016
                                 + 1.18526 w_k1_AXP_s016
                                 + 1.03021 w_k1_DIS_s016
                                 + 1.20949 w_k1_FDX_s016
                                 + 0.864042 w_k1_IBM_s016
                                 + 0.985413 w_k1_INTC_s016
                                 + 0.848964 w_k1_KO_s016
                                 + 0.78945 w_k1_MAT_s016
                                 + 1.0631 w_k1_MSFT_s016 + 1.0858 w_k1_PG_s016
                                 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.3614 w_k1_ADM_s017
                                 + 1.54758 w_k1_AXP_s017 + 1.6768 w_k1_DIS_s017
                                 + 1.45607 w_k1_FDX_s017 + 1.145 w_k1_IBM_s017
                                 + 1.5695 w_k1_INTC_s017 + 1.25774 w_k1_KO_s017
                                 + 0.935606 w_k1_MAT_s017
                                 + 1.46781 w_k1_MSFT_s017
                                 + 1.14181 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.15839 w_k1_ADM_s017
                                 + 0.804635 w_k1_AXP_s017
                                 + 0.940517 w_k1_DIS_s017
                                 + 0.861539 w_k1_FDX_s017
                                 + 1.09497 w_k1_IBM_s017
                                 + 0.979879 w_k1_INTC_s017
                                 + 1.03461 w_k1_KO_s017 + 1.22186 w_k1_MAT_s017
                                 + 0.97511 w_k1_MSFT_s017
                                 + 1.05531 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.81247 w_k1_ADM_s017
                                 + 0.648794 w_k1_AXP_s017
                                 + 0.58567 w_k1_DIS_s017
                                 + 0.760148 w_k1_FDX_s017
                                 + 1.01325 w_k1_IBM_s017
                                 + 0.848491 w_k1_INTC_s017
                                 + 0.75832 w_k1_KO_s017
                                 + 0.786204 w_k1_MAT_s017
                                 + 0.789569 w_k1_MSFT_s017
                                 + 0.857085 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.23655 w_k1_ADM_s017
                                 + 1.05866 w_k1_AXP_s017
                                 + 1.21448 w_k1_DIS_s017
                                 + 1.14602 w_k1_FDX_s017
                                 + 1.07826 w_k1_IBM_s017
                                 + 0.962972 w_k1_INTC_s017
                                 + 1.18197 w_k1_KO_s017 + 1.33167 w_k1_MAT_s017
                                 + 1.3916 w_k1_MSFT_s017 + 1.13654 w_k1_PG_s017
                                 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.11529 w_k1_ADM_s017
                                 + 0.784255 w_k1_AXP_s017
                                 + 0.882726 w_k1_DIS_s017
                                 + 0.904639 w_k1_FDX_s017
                                 + 0.768021 w_k1_IBM_s017
                                 + 0.88554 w_k1_INTC_s017
                                 + 0.798271 w_k1_KO_s017
                                 + 0.590769 w_k1_MAT_s017
                                 + 0.969601 w_k1_MSFT_s017
                                 + 0.870693 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.22715 w_k1_ADM_s017
                                 + 1.25235 w_k1_AXP_s017
                                 + 1.36984 w_k1_DIS_s017 + 1.0448 w_k1_FDX_s017
                                 + 1.69465 w_k1_IBM_s017
                                 + 1.04246 w_k1_INTC_s017
                                 + 1.34135 w_k1_KO_s017 + 2.22102 w_k1_MAT_s017
                                 + 1.2393 w_k1_MSFT_s017 + 1.23373 w_k1_PG_s017
                                 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.33314 w_k1_ADM_s017
                                 + 1.15074 w_k1_AXP_s017
                                 + 1.11911 w_k1_DIS_s017 + 1.2373 w_k1_FDX_s017
                                 + 0.936024 w_k1_IBM_s017
                                 + 1.18247 w_k1_INTC_s017
                                 + 0.959653 w_k1_KO_s017
                                 + 0.994956 w_k1_MAT_s017
                                 + 1.21276 w_k1_MSFT_s017
                                 + 0.984976 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.29952 w_k1_ADM_s017
                                 + 1.08413 w_k1_AXP_s017 + 1.0632 w_k1_DIS_s017
                                 + 1.05112 w_k1_FDX_s017
                                 + 1.33206 w_k1_IBM_s017
                                 + 1.14346 w_k1_INTC_s017
                                 + 1.12461 w_k1_KO_s017 + 0.95421 w_k1_MAT_s017
                                 + 1.03228 w_k1_MSFT_s017
                                 + 1.02621 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.919638 w_k1_ADM_s100
                                 + 1.04377 w_k1_AXP_s100
                                 + 0.737002 w_k1_DIS_s100
                                 + 0.915758 w_k1_FDX_s100
                                 + 1.28295 w_k1_IBM_s100
                                 + 0.921166 w_k1_INTC_s100
                                 + 0.889424 w_k1_KO_s100
                                 + 1.25372 w_k1_MAT_s100
                                 + 0.878991 w_k1_MSFT_s100
                                 + 0.932174 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.19311 w_k1_ADM_s100
                                 + 1.19203 w_k1_AXP_s100
                                 + 0.903489 w_k1_DIS_s100
                                 + 1.06382 w_k1_FDX_s100
                                 + 0.898879 w_k1_IBM_s100
                                 + 1.00531 w_k1_INTC_s100
                                 + 0.861659 w_k1_KO_s100
                                 + 0.786885 w_k1_MAT_s100
                                 + 0.876871 w_k1_MSFT_s100
                                 + 0.919806 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.890184 w_k1_ADM_s100
                                 + 0.674969 w_k1_AXP_s100
                                 + 0.617088 w_k1_DIS_s100
                                 + 0.805596 w_k1_FDX_s100
                                 + 0.769806 w_k1_IBM_s100
                                 + 0.78878 w_k1_INTC_s100
                                 + 0.709358 w_k1_KO_s100
                                 + 0.697623 w_k1_MAT_s100
                                 + 0.718833 w_k1_MSFT_s100
                                 + 0.767109 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.03938 w_k1_ADM_s100
                                 + 0.791647 w_k1_AXP_s100
                                 + 1.00957 w_k1_DIS_s100
                                 + 0.952378 w_k1_FDX_s100
                                 + 1.09589 w_k1_IBM_s100
                                 + 1.04664 w_k1_INTC_s100
                                 + 1.10891 w_k1_KO_s100
                                 + 0.731709 w_k1_MAT_s100
                                 + 0.936262 w_k1_MSFT_s100 + 1.004 w_k1_PG_s100
                                 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.10806 w_k1_ADM_s100
                                 + 0.713146 w_k1_AXP_s100
                                 + 1.04466 w_k1_DIS_s100
                                 + 0.982154 w_k1_FDX_s100
                                 + 0.815388 w_k1_IBM_s100
                                 + 1.08701 w_k1_INTC_s100
                                 + 0.961338 w_k1_KO_s100
                                 + 0.603898 w_k1_MAT_s100
                                 + 0.964673 w_k1_MSFT_s100
                                 + 0.923659 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.19331 w_k1_ADM_s100
                                 + 1.07104 w_k1_AXP_s100
                                 + 1.13008 w_k1_DIS_s100
                                 + 1.15042 w_k1_FDX_s100
                                 + 1.15476 w_k1_IBM_s100
                                 + 0.97412 w_k1_INTC_s100
                                 + 1.06087 w_k1_KO_s100 + 1.24157 w_k1_MAT_s100
                                 + 0.984002 w_k1_MSFT_s100
                                 + 1.19618 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.4294 w_k1_ADM_s100
                                 + 1.1633 w_k1_AXP_s100 + 1.49297 w_k1_DIS_s100
                                 + 1.29406 w_k1_FDX_s100
                                 + 1.02609 w_k1_IBM_s100
                                 + 0.988567 w_k1_INTC_s100
                                 + 1.1685 w_k1_KO_s100 + 1.16038 w_k1_MAT_s100
                                 + 1.10828 w_k1_MSFT_s100
                                 + 1.25872 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.35361 w_k1_ADM_s100
                                 + 1.33713 w_k1_AXP_s100
                                 + 1.34879 w_k1_DIS_s100
                                 + 1.10256 w_k1_FDX_s100
                                 + 1.43431 w_k1_IBM_s100
                                 + 1.17101 w_k1_INTC_s100
                                 + 1.28727 w_k1_KO_s100 + 1.47472 w_k1_MAT_s100
                                 + 1.10283 w_k1_MSFT_s100
                                 + 1.17772 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.624382 w_k1_ADM_s101
                                 + 0.580665 w_k1_AXP_s101
                                 + 0.779103 w_k1_DIS_s101
                                 + 0.700867 w_k1_FDX_s101
                                 + 0.933245 w_k1_IBM_s101
                                 + 0.801121 w_k1_INTC_s101
                                 + 0.807886 w_k1_KO_s101
                                 + 0.743451 w_k1_MAT_s101
                                 + 0.855792 w_k1_MSFT_s101
                                 + 0.87434 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.86116 w_k1_ADM_s101
                                 + 1.06586 w_k1_AXP_s101
                                 + 0.83693 w_k1_DIS_s101
                                 + 0.820119 w_k1_FDX_s101
                                 + 0.932501 w_k1_IBM_s101
                                 + 0.822773 w_k1_INTC_s101
                                 + 0.899917 w_k1_KO_s101
                                 + 1.13483 w_k1_MAT_s101
                                 + 0.82355 w_k1_MSFT_s101 + 1.0111 w_k1_PG_s101
                                 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.930085 w_k1_ADM_s101
                                 + 0.756887 w_k1_AXP_s101
                                 + 0.863533 w_k1_DIS_s101
                                 + 0.899441 w_k1_FDX_s101
                                 + 0.743136 w_k1_IBM_s101
                                 + 0.856751 w_k1_INTC_s101
                                 + 0.888648 w_k1_KO_s101
                                 + 0.685627 w_k1_MAT_s101
                                 + 0.95947 w_k1_MSFT_s101
                                 + 1.02471 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.02541 w_k1_ADM_s101
                                 + 1.20256 w_k1_AXP_s101
                                 + 1.24613 w_k1_DIS_s101
                                 + 1.11157 w_k1_FDX_s101
                                 + 0.964174 w_k1_IBM_s101
                                 + 1.08106 w_k1_INTC_s101
                                 + 1.13141 w_k1_KO_s101
                                 + 0.915354 w_k1_MAT_s101
                                 + 1.13861 w_k1_MSFT_s101
                                 + 1.16049 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.10945 w_k1_ADM_s101
                                 + 1.26324 w_k1_AXP_s101
                                 + 1.16133 w_k1_DIS_s101
                                 + 1.10861 w_k1_FDX_s101
                                 + 1.22721 w_k1_IBM_s101
                                 + 1.01515 w_k1_INTC_s101
                                 + 1.25657 w_k1_KO_s101 + 1.40261 w_k1_MAT_s101
                                 + 0.912958 w_k1_MSFT_s101
                                 + 1.18065 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.919335 w_k1_ADM_s101
                                 + 1.247 w_k1_AXP_s101 + 1.33972 w_k1_DIS_s101
                                 + 1.0638 w_k1_FDX_s101 + 1.59043 w_k1_IBM_s101
                                 + 0.876975 w_k1_INTC_s101
                                 + 1.33762 w_k1_KO_s101 + 1.72952 w_k1_MAT_s101
                                 + 0.926169 w_k1_MSFT_s101
                                 + 1.14245 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.32663 w_k1_ADM_s101
                                 + 1.328 w_k1_AXP_s101 + 1.85945 w_k1_DIS_s101
                                 + 1.38159 w_k1_FDX_s101
                                 + 1.10717 w_k1_IBM_s101
                                 + 1.42195 w_k1_INTC_s101
                                 + 1.48071 w_k1_KO_s101 + 1.06988 w_k1_MAT_s101
                                 + 1.35078 w_k1_MSFT_s101
                                 + 1.39588 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.87735 w_k1_ADM_s101
                                 + 0.701735 w_k1_AXP_s101
                                 + 1.26004 w_k1_DIS_s101
                                 + 0.890343 w_k1_FDX_s101
                                 + 1.17143 w_k1_IBM_s101
                                 + 1.07354 w_k1_INTC_s101
                                 + 1.27111 w_k1_KO_s101
                                 + 0.837647 w_k1_MAT_s101
                                 + 1.05816 w_k1_MSFT_s101
                                 + 1.02343 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.22053 w_k1_ADM_s102
                                 + 1.36496 w_k1_AXP_s102
                                 + 1.45855 w_k1_DIS_s102
                                 + 1.45585 w_k1_FDX_s102
                                 + 0.813837 w_k1_IBM_s102
                                 + 1.47662 w_k1_INTC_s102
                                 + 0.96458 w_k1_KO_s102
                                 + 0.841088 w_k1_MAT_s102
                                 + 1.33475 w_k1_MSFT_s102
                                 + 1.09095 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.12036 w_k1_ADM_s102
                                 + 0.903211 w_k1_AXP_s102
                                 + 0.947935 w_k1_DIS_s102
                                 + 1.02075 w_k1_FDX_s102
                                 + 0.875304 w_k1_IBM_s102
                                 + 1.03692 w_k1_INTC_s102
                                 + 0.958695 w_k1_KO_s102
                                 + 0.605509 w_k1_MAT_s102
                                 + 1.07311 w_k1_MSFT_s102
                                 + 1.04514 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.13102 w_k1_ADM_s102
                                 + 1.17204 w_k1_AXP_s102
                                 + 1.16217 w_k1_DIS_s102
                                 + 1.01241 w_k1_FDX_s102 + 1.5929 w_k1_IBM_s102
                                 + 1.20641 w_k1_INTC_s102
                                 + 1.13094 w_k1_KO_s102 + 1.30271 w_k1_MAT_s102
                                 + 1.10638 w_k1_MSFT_s102
                                 + 1.08164 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.744469 w_k1_ADM_s102
                                 + 0.672981 w_k1_AXP_s102
                                 + 0.642594 w_k1_DIS_s102
                                 + 0.792763 w_k1_FDX_s102
                                 + 0.748631 w_k1_IBM_s102
                                 + 0.875251 w_k1_INTC_s102
                                 + 0.68987 w_k1_KO_s102
                                 + 0.639617 w_k1_MAT_s102
                                 + 0.816798 w_k1_MSFT_s102
                                 + 0.825534 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.00887 w_k1_ADM_s102
                                 + 0.828671 w_k1_AXP_s102
                                 + 0.733268 w_k1_DIS_s102
                                 + 0.853189 w_k1_FDX_s102
                                 + 1.11513 w_k1_IBM_s102
                                 + 0.786741 w_k1_INTC_s102
                                 + 0.771902 w_k1_KO_s102
                                 + 0.979769 w_k1_MAT_s102
                                 + 0.797095 w_k1_MSFT_s102
                                 + 1.01405 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.25067 w_k1_ADM_s102
                                 + 1.06739 w_k1_AXP_s102
                                 + 1.29739 w_k1_DIS_s102
                                 + 1.10228 w_k1_FDX_s102
                                 + 1.10764 w_k1_IBM_s102
                                 + 1.21031 w_k1_INTC_s102
                                 + 1.11318 w_k1_KO_s102
                                 + 0.907361 w_k1_MAT_s102
                                 + 1.12818 w_k1_MSFT_s102
                                 + 1.22766 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.08362 w_k1_ADM_s102
                                 + 1.0135 w_k1_AXP_s102 + 1.00788 w_k1_DIS_s102
                                 + 1.24433 w_k1_FDX_s102
                                 + 0.602534 w_k1_IBM_s102
                                 + 1.19315 w_k1_INTC_s102
                                 + 0.851372 w_k1_KO_s102
                                 + 0.802086 w_k1_MAT_s102
                                 + 1.16136 w_k1_MSFT_s102
                                 + 1.09189 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.07368 w_k1_ADM_s102
                                 + 0.917424 w_k1_AXP_s102
                                 + 0.878437 w_k1_DIS_s102
                                 + 1.0663 w_k1_FDX_s102
                                 + 0.566243 w_k1_IBM_s102
                                 + 1.06005 w_k1_INTC_s102
                                 + 0.744622 w_k1_KO_s102
                                 + 0.668806 w_k1_MAT_s102
                                 + 1.11202 w_k1_MSFT_s102
                                 + 0.986265 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.02644 w_k1_ADM_s103
                                 + 1.18179 w_k1_AXP_s103
                                 + 1.20264 w_k1_DIS_s103
                                 + 1.12882 w_k1_FDX_s103
                                 + 0.85625 w_k1_IBM_s103
                                 + 1.03286 w_k1_INTC_s103
                                 + 0.913411 w_k1_KO_s103
                                 + 0.961979 w_k1_MAT_s103
                                 + 0.986947 w_k1_MSFT_s103
                                 + 1.09534 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.17805 w_k1_ADM_s103
                                 + 0.998934 w_k1_AXP_s103
                                 + 0.860264 w_k1_DIS_s103
                                 + 1.08419 w_k1_FDX_s103
                                 + 0.771041 w_k1_IBM_s103
                                 + 0.963571 w_k1_INTC_s103
                                 + 0.853203 w_k1_KO_s103
                                 + 0.977453 w_k1_MAT_s103
                                 + 1.03171 w_k1_MSFT_s103
                                 + 1.07142 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.908977 w_k1_ADM_s103
                                 + 0.822266 w_k1_AXP_s103
                                 + 0.883255 w_k1_DIS_s103
                                 + 0.975893 w_k1_FDX_s103
                                 + 0.795225 w_k1_IBM_s103
                                 + 1.07257 w_k1_INTC_s103
                                 + 0.819453 w_k1_KO_s103
                                 + 0.733726 w_k1_MAT_s103
                                 + 1.00721 w_k1_MSFT_s103
                                 + 0.915564 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.746172 w_k1_ADM_s103
                                 + 0.695592 w_k1_AXP_s103
                                 + 0.646982 w_k1_DIS_s103
                                 + 0.715087 w_k1_FDX_s103
                                 + 0.81016 w_k1_IBM_s103
                                 + 0.77438 w_k1_INTC_s103
                                 + 0.725975 w_k1_KO_s103
                                 + 0.773619 w_k1_MAT_s103
                                 + 0.663035 w_k1_MSFT_s103
                                 + 0.811994 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.58789 w_k1_ADM_s103
                                 + 1.53608 w_k1_AXP_s103
                                 + 1.70636 w_k1_DIS_s103
                                 + 1.35065 w_k1_FDX_s103
                                 + 1.11227 w_k1_IBM_s103
                                 + 1.34111 w_k1_INTC_s103 + 1.1596 w_k1_KO_s103
                                 + 1.29623 w_k1_MAT_s103
                                 + 1.51016 w_k1_MSFT_s103
                                 + 1.43494 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.0591 w_k1_ADM_s103
                                 + 1.05556 w_k1_AXP_s103
                                 + 1.21513 w_k1_DIS_s103
                                 + 0.984835 w_k1_FDX_s103
                                 + 1.14272 w_k1_IBM_s103
                                 + 1.08959 w_k1_INTC_s103
                                 + 1.23869 w_k1_KO_s103 + 1.17827 w_k1_MAT_s103
                                 + 1.01469 w_k1_MSFT_s103
                                 + 1.13082 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.898792 w_k1_ADM_s103
                                 + 1.09944 w_k1_AXP_s103
                                 + 0.952632 w_k1_DIS_s103
                                 + 0.843879 w_k1_FDX_s103
                                 + 1.4834 w_k1_IBM_s103
                                 + 0.755437 w_k1_INTC_s103
                                 + 1.14822 w_k1_KO_s103 + 1.67692 w_k1_MAT_s103
                                 + 0.835767 w_k1_MSFT_s103
                                 + 1.06394 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.909583 w_k1_ADM_s103
                                 + 0.953118 w_k1_AXP_s103
                                 + 0.71807 w_k1_DIS_s103
                                 + 0.801484 w_k1_FDX_s103
                                 + 0.926705 w_k1_IBM_s103
                                 + 0.89262 w_k1_INTC_s103
                                 + 0.797525 w_k1_KO_s103
                                 + 0.996881 w_k1_MAT_s103
                                 + 0.823663 w_k1_MSFT_s103
                                 + 0.981777 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.07321 w_k1_ADM_s104
                                 + 0.985327 w_k1_AXP_s104
                                 + 0.931987 w_k1_DIS_s104
                                 + 1.07177 w_k1_FDX_s104
                                 + 0.714987 w_k1_IBM_s104
                                 + 1.03619 w_k1_INTC_s104
                                 + 0.823069 w_k1_KO_s104
                                 + 0.756124 w_k1_MAT_s104
                                 + 0.911883 w_k1_MSFT_s104
                                 + 0.973025 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.00669 w_k1_ADM_s104
                                 + 0.829311 w_k1_AXP_s104
                                 + 1.1402 w_k1_DIS_s104 + 1.00502 w_k1_FDX_s104
                                 + 0.70512 w_k1_IBM_s104
                                 + 0.905729 w_k1_INTC_s104
                                 + 0.850804 w_k1_KO_s104
                                 + 0.55755 w_k1_MAT_s104
                                 + 1.08541 w_k1_MSFT_s104
                                 + 0.87243 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.691819 w_k1_ADM_s104
                                 + 0.627919 w_k1_AXP_s104
                                 + 0.505914 w_k1_DIS_s104
                                 + 0.733597 w_k1_FDX_s104
                                 + 0.601468 w_k1_IBM_s104
                                 + 0.790172 w_k1_INTC_s104
                                 + 0.659455 w_k1_KO_s104
                                 + 0.637156 w_k1_MAT_s104
                                 + 0.755498 w_k1_MSFT_s104
                                 + 0.803866 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.781538 w_k1_ADM_s104
                                 + 0.728351 w_k1_AXP_s104
                                 + 0.702347 w_k1_DIS_s104
                                 + 0.816249 w_k1_FDX_s104
                                 + 0.762971 w_k1_IBM_s104
                                 + 0.649123 w_k1_INTC_s104
                                 + 0.719823 w_k1_KO_s104
                                 + 0.758376 w_k1_MAT_s104
                                 + 0.710529 w_k1_MSFT_s104
                                 + 0.853206 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.782061 w_k1_ADM_s104
                                 + 1.08637 w_k1_AXP_s104
                                 + 0.998147 w_k1_DIS_s104
                                 + 0.967866 w_k1_FDX_s104
                                 + 1.16104 w_k1_IBM_s104
                                 + 0.980908 w_k1_INTC_s104
                                 + 1.00784 w_k1_KO_s104 + 1.19004 w_k1_MAT_s104
                                 + 0.985082 w_k1_MSFT_s104
                                 + 1.0382 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.15744 w_k1_ADM_s104
                                 + 1.59432 w_k1_AXP_s104
                                 + 1.61434 w_k1_DIS_s104
                                 + 1.34028 w_k1_FDX_s104
                                 + 1.66483 w_k1_IBM_s104
                                 + 1.4115 w_k1_INTC_s104 + 1.47605 w_k1_KO_s104
                                 + 2.07627 w_k1_MAT_s104
                                 + 1.30809 w_k1_MSFT_s104
                                 + 1.37046 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.85529 w_k1_ADM_s104
                                 + 1.27201 w_k1_AXP_s104
                                 + 1.12239 w_k1_DIS_s104
                                 + 0.966338 w_k1_FDX_s104
                                 + 1.58512 w_k1_IBM_s104
                                 + 1.00639 w_k1_INTC_s104
                                 + 1.28371 w_k1_KO_s104 + 1.60647 w_k1_MAT_s104
                                 + 0.963009 w_k1_MSFT_s104
                                 + 1.11414 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.861018 w_k1_ADM_s104
                                 + 0.932511 w_k1_AXP_s104
                                 + 0.816224 w_k1_DIS_s104
                                 + 0.889175 w_k1_FDX_s104
                                 + 0.905286 w_k1_IBM_s104
                                 + 0.910478 w_k1_INTC_s104
                                 + 0.870697 w_k1_KO_s104
                                 + 0.912257 w_k1_MAT_s104
                                 + 0.86669 w_k1_MSFT_s104
                                 + 0.883052 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.968135 w_k1_ADM_s105
                                 + 1.01036 w_k1_AXP_s105
                                 + 0.883488 w_k1_DIS_s105
                                 + 0.911992 w_k1_FDX_s105
                                 + 1.09281 w_k1_IBM_s105
                                 + 0.848097 w_k1_INTC_s105
                                 + 0.870113 w_k1_KO_s105
                                 + 0.982467 w_k1_MAT_s105
                                 + 0.905158 w_k1_MSFT_s105
                                 + 0.838848 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.21114 w_k1_ADM_s105
                                 + 1.47893 w_k1_AXP_s105 + 1.1708 w_k1_DIS_s105
                                 + 1.08337 w_k1_FDX_s105
                                 + 1.58266 w_k1_IBM_s105
                                 + 1.02309 w_k1_INTC_s105
                                 + 1.07942 w_k1_KO_s105 + 1.32057 w_k1_MAT_s105
                                 + 1.03969 w_k1_MSFT_s105
                                 + 1.04557 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.811366 w_k1_ADM_s105
                                 + 0.837301 w_k1_AXP_s105
                                 + 0.873159 w_k1_DIS_s105
                                 + 0.812771 w_k1_FDX_s105
                                 + 1.43804 w_k1_IBM_s105
                                 + 0.592373 w_k1_INTC_s105
                                 + 1.19504 w_k1_KO_s105 + 1.22087 w_k1_MAT_s105
                                 + 0.934698 w_k1_MSFT_s105
                                 + 0.968673 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.786297 w_k1_ADM_s105
                                 + 0.685125 w_k1_AXP_s105
                                 + 0.677878 w_k1_DIS_s105
                                 + 0.772742 w_k1_FDX_s105
                                 + 0.786776 w_k1_IBM_s105
                                 + 0.756689 w_k1_INTC_s105
                                 + 0.844436 w_k1_KO_s105
                                 + 0.677542 w_k1_MAT_s105
                                 + 0.819769 w_k1_MSFT_s105
                                 + 0.756604 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.19505 w_k1_ADM_s105
                                 + 1.22515 w_k1_AXP_s105
                                 + 1.69988 w_k1_DIS_s105
                                 + 1.34431 w_k1_FDX_s105
                                 + 1.32738 w_k1_IBM_s105
                                 + 1.10246 w_k1_INTC_s105
                                 + 1.43939 w_k1_KO_s105 + 1.38648 w_k1_MAT_s105
                                 + 1.25065 w_k1_MSFT_s105
                                 + 1.28266 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.10423 w_k1_ADM_s105
                                 + 1.07123 w_k1_AXP_s105
                                 + 1.18258 w_k1_DIS_s105
                                 + 1.25974 w_k1_FDX_s105
                                 + 0.737904 w_k1_IBM_s105
                                 + 1.19747 w_k1_INTC_s105
                                 + 0.848395 w_k1_KO_s105
                                 + 0.80658 w_k1_MAT_s105
                                 + 1.17521 w_k1_MSFT_s105
                                 + 0.977732 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.0532 w_k1_ADM_s105
                                 + 0.953566 w_k1_AXP_s105
                                 + 0.92582 w_k1_DIS_s105
                                 + 0.986511 w_k1_FDX_s105
                                 + 0.754332 w_k1_IBM_s105
                                 + 0.814979 w_k1_INTC_s105
                                 + 0.880093 w_k1_KO_s105
                                 + 0.53237 w_k1_MAT_s105
                                 + 0.869604 w_k1_MSFT_s105
                                 + 0.918982 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.26604 w_k1_ADM_s105
                                 + 1.17505 w_k1_AXP_s105
                                 + 1.35214 w_k1_DIS_s105
                                 + 1.11335 w_k1_FDX_s105
                                 + 1.64008 w_k1_IBM_s105
                                 + 0.912023 w_k1_INTC_s105
                                 + 1.27967 w_k1_KO_s105 + 1.80478 w_k1_MAT_s105
                                 + 1.16003 w_k1_MSFT_s105
                                 + 1.14242 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.6418 w_k1_ADM_s106
                                 + 0.737155 w_k1_AXP_s106
                                 + 0.717272 w_k1_DIS_s106
                                 + 0.81823 w_k1_FDX_s106
                                 + 0.76981 w_k1_IBM_s106
                                 + 0.923326 w_k1_INTC_s106
                                 + 0.797543 w_k1_KO_s106
                                 + 0.747962 w_k1_MAT_s106
                                 + 1.02296 w_k1_MSFT_s106
                                 + 0.790927 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.02169 w_k1_ADM_s106
                                 + 1.01253 w_k1_AXP_s106
                                 + 0.998391 w_k1_DIS_s106
                                 + 1.12564 w_k1_FDX_s106
                                 + 0.799784 w_k1_IBM_s106
                                 + 1.24915 w_k1_INTC_s106
                                 + 0.972399 w_k1_KO_s106
                                 + 0.819228 w_k1_MAT_s106
                                 + 1.19563 w_k1_MSFT_s106
                                 + 1.12931 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.14078 w_k1_ADM_s106
                                 + 1.34487 w_k1_AXP_s106 + 1.4412 w_k1_DIS_s106
                                 + 1.3292 w_k1_FDX_s106 + 1.1214 w_k1_IBM_s106
                                 + 1.43966 w_k1_INTC_s106
                                 + 1.19542 w_k1_KO_s106 + 1.0403 w_k1_MAT_s106
                                 + 1.27087 w_k1_MSFT_s106
                                 + 1.14387 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.882106 w_k1_ADM_s106
                                 + 1.19195 w_k1_AXP_s106
                                 + 1.24265 w_k1_DIS_s106
                                 + 1.11205 w_k1_FDX_s106
                                 + 1.95293 w_k1_IBM_s106
                                 + 1.07573 w_k1_INTC_s106
                                 + 1.56383 w_k1_KO_s106 + 1.94483 w_k1_MAT_s106
                                 + 1.10666 w_k1_MSFT_s106
                                 + 1.28852 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.580572 w_k1_ADM_s106
                                 + 0.929648 w_k1_AXP_s106
                                 + 0.898234 w_k1_DIS_s106
                                 + 0.841318 w_k1_FDX_s106
                                 + 0.999725 w_k1_IBM_s106
                                 + 1.00746 w_k1_INTC_s106
                                 + 1.06497 w_k1_KO_s106 + 1.078 w_k1_MAT_s106
                                 + 0.996483 w_k1_MSFT_s106
                                 + 0.973394 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.04732 w_k1_ADM_s106
                                 + 1.21015 w_k1_AXP_s106
                                 + 0.915966 w_k1_DIS_s106
                                 + 0.966223 w_k1_FDX_s106
                                 + 1.39638 w_k1_IBM_s106
                                 + 1.0801 w_k1_INTC_s106 + 1.13537 w_k1_KO_s106
                                 + 1.41969 w_k1_MAT_s106
                                 + 1.14376 w_k1_MSFT_s106
                                 + 1.08157 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.775378 w_k1_ADM_s106
                                 + 0.665796 w_k1_AXP_s106
                                 + 0.552264 w_k1_DIS_s106
                                 + 0.762084 w_k1_FDX_s106
                                 + 0.560341 w_k1_IBM_s106
                                 + 0.857781 w_k1_INTC_s106
                                 + 0.650767 w_k1_KO_s106
                                 + 0.455579 w_k1_MAT_s106
                                 + 0.764481 w_k1_MSFT_s106
                                 + 0.748945 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.867482 w_k1_ADM_s106
                                 + 1.02678 w_k1_AXP_s106
                                 + 0.811204 w_k1_DIS_s106
                                 + 0.942343 w_k1_FDX_s106
                                 + 1.09286 w_k1_IBM_s106
                                 + 1.34734 w_k1_INTC_s106
                                 + 0.987575 w_k1_KO_s106
                                 + 0.839879 w_k1_MAT_s106
                                 + 1.02814 w_k1_MSFT_s106
                                 + 0.936597 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.06418 w_k1_ADM_s107
                                 + 1.07198 w_k1_AXP_s107
                                 + 0.988714 w_k1_DIS_s107
                                 + 0.980134 w_k1_FDX_s107
                                 + 1.12496 w_k1_IBM_s107
                                 + 0.923188 w_k1_INTC_s107
                                 + 1.0468 w_k1_KO_s107 + 1.19509 w_k1_MAT_s107
                                 + 0.949824 w_k1_MSFT_s107
                                 + 1.06831 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.20761 w_k1_ADM_s107
                                 + 1.58595 w_k1_AXP_s107
                                 + 1.62756 w_k1_DIS_s107
                                 + 1.21019 w_k1_FDX_s107
                                 + 1.45534 w_k1_IBM_s107
                                 + 1.36538 w_k1_INTC_s107
                                 + 1.43603 w_k1_KO_s107 + 2.21762 w_k1_MAT_s107
                                 + 1.31048 w_k1_MSFT_s107
                                 + 1.31719 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.36776 w_k1_ADM_s107
                                 + 1.30419 w_k1_AXP_s107
                                 + 1.62541 w_k1_DIS_s107
                                 + 1.40827 w_k1_FDX_s107
                                 + 0.93892 w_k1_IBM_s107
                                 + 1.28953 w_k1_INTC_s107
                                 + 1.18286 w_k1_KO_s107 + 1.37087 w_k1_MAT_s107
                                 + 1.32418 w_k1_MSFT_s107
                                 + 1.18578 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.26507 w_k1_ADM_s107
                                 + 1.20617 w_k1_AXP_s107
                                 + 1.00129 w_k1_DIS_s107
                                 + 1.20238 w_k1_FDX_s107
                                 + 0.813351 w_k1_IBM_s107
                                 + 1.25555 w_k1_INTC_s107
                                 + 0.841628 w_k1_KO_s107
                                 + 1.13331 w_k1_MAT_s107
                                 + 1.11376 w_k1_MSFT_s107
                                 + 1.02785 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.810949 w_k1_ADM_s107
                                 + 0.829033 w_k1_AXP_s107
                                 + 0.785393 w_k1_DIS_s107
                                 + 0.727115 w_k1_FDX_s107
                                 + 1.22516 w_k1_IBM_s107
                                 + 0.809096 w_k1_INTC_s107
                                 + 0.928092 w_k1_KO_s107
                                 + 1.23049 w_k1_MAT_s107
                                 + 0.752544 w_k1_MSFT_s107
                                 + 0.989533 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.877072 w_k1_ADM_s107
                                 + 0.918177 w_k1_AXP_s107
                                 + 0.72085 w_k1_DIS_s107
                                 + 0.905056 w_k1_FDX_s107
                                 + 0.782301 w_k1_IBM_s107
                                 + 0.979841 w_k1_INTC_s107
                                 + 0.748071 w_k1_KO_s107
                                 + 0.87566 w_k1_MAT_s107
                                 + 0.84031 w_k1_MSFT_s107
                                 + 0.894378 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.23924 w_k1_ADM_s107
                                 + 1.3337 w_k1_AXP_s107 + 1.07139 w_k1_DIS_s107
                                 + 1.06386 w_k1_FDX_s107
                                 + 1.48639 w_k1_IBM_s107
                                 + 1.13439 w_k1_INTC_s107
                                 + 1.09861 w_k1_KO_s107 + 1.6805 w_k1_MAT_s107
                                 + 1.01466 w_k1_MSFT_s107
                                 + 1.04146 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.85133 w_k1_ADM_s107
                                 + 0.842606 w_k1_AXP_s107
                                 + 0.639202 w_k1_DIS_s107
                                 + 0.705554 w_k1_FDX_s107
                                 + 0.775789 w_k1_IBM_s107
                                 + 0.775216 w_k1_INTC_s107
                                 + 0.681036 w_k1_KO_s107
                                 + 0.751828 w_k1_MAT_s107
                                 + 0.618214 w_k1_MSFT_s107
                                 + 0.768069 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.03195 w_k1_ADM_s110
                                 + 1.13951 w_k1_AXP_s110
                                 + 1.35763 w_k1_DIS_s110
                                 + 1.37734 w_k1_FDX_s110
                                 + 0.834165 w_k1_IBM_s110
                                 + 1.34039 w_k1_INTC_s110
                                 + 0.97382 w_k1_KO_s110
                                 + 0.769094 w_k1_MAT_s110
                                 + 1.25099 w_k1_MSFT_s110 + 1.0329 w_k1_PG_s110
                                 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.12485 w_k1_ADM_s110
                                 + 0.882412 w_k1_AXP_s110
                                 + 1.04423 w_k1_DIS_s110
                                 + 1.06302 w_k1_FDX_s110
                                 + 0.856279 w_k1_IBM_s110
                                 + 1.00931 w_k1_INTC_s110
                                 + 1.10614 w_k1_KO_s110 + 0.79295 w_k1_MAT_s110
                                 + 1.09907 w_k1_MSFT_s110
                                 + 0.984799 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.26482 w_k1_ADM_s110
                                 + 1.41704 w_k1_AXP_s110
                                 + 1.40229 w_k1_DIS_s110
                                 + 1.27997 w_k1_FDX_s110
                                 + 0.977132 w_k1_IBM_s110
                                 + 1.51573 w_k1_INTC_s110
                                 + 1.05942 w_k1_KO_s110
                                 + 0.940942 w_k1_MAT_s110
                                 + 1.1725 w_k1_MSFT_s110 + 1.06052 w_k1_PG_s110
                                 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.834691 w_k1_ADM_s110
                                 + 0.770138 w_k1_AXP_s110
                                 + 0.730782 w_k1_DIS_s110
                                 + 0.826091 w_k1_FDX_s110
                                 + 0.759895 w_k1_IBM_s110
                                 + 0.840585 w_k1_INTC_s110
                                 + 0.664422 w_k1_KO_s110
                                 + 0.535092 w_k1_MAT_s110
                                 + 0.790215 w_k1_MSFT_s110
                                 + 0.750053 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.10719 w_k1_ADM_s110
                                 + 0.997674 w_k1_AXP_s110
                                 + 1.29588 w_k1_DIS_s110
                                 + 1.10369 w_k1_FDX_s110
                                 + 1.38829 w_k1_IBM_s110
                                 + 1.35961 w_k1_INTC_s110
                                 + 1.42823 w_k1_KO_s110
                                 + 0.959967 w_k1_MAT_s110
                                 + 1.15355 w_k1_MSFT_s110
                                 + 0.965859 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.959821 w_k1_ADM_s110
                                 + 1.35462 w_k1_AXP_s110
                                 + 0.965719 w_k1_DIS_s110
                                 + 1.01857 w_k1_FDX_s110
                                 + 1.64636 w_k1_IBM_s110
                                 + 1.15907 w_k1_INTC_s110
                                 + 1.17261 w_k1_KO_s110 + 1.2046 w_k1_MAT_s110
                                 + 0.957304 w_k1_MSFT_s110
                                 + 1.03825 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.877142 w_k1_ADM_s110
                                 + 0.723142 w_k1_AXP_s110
                                 + 0.932041 w_k1_DIS_s110
                                 + 0.880792 w_k1_FDX_s110
                                 + 0.960506 w_k1_IBM_s110
                                 + 1.08204 w_k1_INTC_s110
                                 + 0.830179 w_k1_KO_s110
                                 + 0.560057 w_k1_MAT_s110
                                 + 1.06967 w_k1_MSFT_s110
                                 + 0.838771 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.01366 w_k1_ADM_s110
                                 + 0.979981 w_k1_AXP_s110
                                 + 0.885276 w_k1_DIS_s110
                                 + 1.06463 w_k1_FDX_s110
                                 + 0.898983 w_k1_IBM_s110
                                 + 1.02583 w_k1_INTC_s110
                                 + 0.816095 w_k1_KO_s110
                                 + 0.921807 w_k1_MAT_s110
                                 + 0.96009 w_k1_MSFT_s110
                                 + 0.947085 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.2974 w_k1_ADM_s111
                                 + 1.57662 w_k1_AXP_s111
                                 + 1.51363 w_k1_DIS_s111
                                 + 1.31284 w_k1_FDX_s111
                                 + 1.63384 w_k1_IBM_s111
                                 + 1.23696 w_k1_INTC_s111
                                 + 1.60125 w_k1_KO_s111 + 1.96711 w_k1_MAT_s111
                                 + 1.22448 w_k1_MSFT_s111
                                 + 1.21739 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.01117 w_k1_ADM_s111
                                 + 0.755507 w_k1_AXP_s111
                                 + 0.71128 w_k1_DIS_s111
                                 + 0.882928 w_k1_FDX_s111
                                 + 0.756459 w_k1_IBM_s111
                                 + 0.830846 w_k1_INTC_s111
                                 + 0.98969 w_k1_KO_s111
                                 + 0.784932 w_k1_MAT_s111
                                 + 0.79309 w_k1_MSFT_s111
                                 + 0.755576 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.07574 w_k1_ADM_s111
                                 + 0.906086 w_k1_AXP_s111
                                 + 0.996351 w_k1_DIS_s111
                                 + 1.12795 w_k1_FDX_s111
                                 + 0.686585 w_k1_IBM_s111
                                 + 1.00378 w_k1_INTC_s111
                                 + 0.919164 w_k1_KO_s111
                                 + 0.530784 w_k1_MAT_s111
                                 + 0.994592 w_k1_MSFT_s111
                                 + 0.871173 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.869748 w_k1_ADM_s111
                                 + 0.743591 w_k1_AXP_s111
                                 + 0.663654 w_k1_DIS_s111
                                 + 0.898652 w_k1_FDX_s111
                                 + 0.425018 w_k1_IBM_s111
                                 + 1.06057 w_k1_INTC_s111
                                 + 0.699404 w_k1_KO_s111
                                 + 0.353659 w_k1_MAT_s111
                                 + 0.834127 w_k1_MSFT_s111
                                 + 0.755551 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.0519 w_k1_ADM_s111
                                 + 1.25157 w_k1_AXP_s111
                                 + 0.98325 w_k1_DIS_s111 + 1.0059 w_k1_FDX_s111
                                 + 1.08578 w_k1_IBM_s111
                                 + 0.987344 w_k1_INTC_s111
                                 + 1.04886 w_k1_KO_s111
                                 + 0.902752 w_k1_MAT_s111
                                 + 0.885762 w_k1_MSFT_s111
                                 + 0.88837 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.12474 w_k1_ADM_s111
                                 + 1.15707 w_k1_AXP_s111
                                 + 1.50138 w_k1_DIS_s111 + 1.4272 w_k1_FDX_s111
                                 + 0.891956 w_k1_IBM_s111
                                 + 1.35542 w_k1_INTC_s111
                                 + 1.26269 w_k1_KO_s111 + 0.86206 w_k1_MAT_s111
                                 + 1.3035 w_k1_MSFT_s111 + 1.15081 w_k1_PG_s111
                                 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.853138 w_k1_ADM_s111
                                 + 0.796658 w_k1_AXP_s111
                                 + 0.78143 w_k1_DIS_s111
                                 + 0.900053 w_k1_FDX_s111
                                 + 1.4379 w_k1_IBM_s111
                                 + 0.952566 w_k1_INTC_s111
                                 + 1.16141 w_k1_KO_s111 + 1.12025 w_k1_MAT_s111
                                 + 0.880718 w_k1_MSFT_s111
                                 + 0.918706 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.04568 w_k1_ADM_s111
                                 + 1.09208 w_k1_AXP_s111
                                 + 1.28239 w_k1_DIS_s111
                                 + 1.16458 w_k1_FDX_s111
                                 + 1.15182 w_k1_IBM_s111
                                 + 1.27284 w_k1_INTC_s111
                                 + 1.16858 w_k1_KO_s111
                                 + 0.894629 w_k1_MAT_s111
                                 + 1.14617 w_k1_MSFT_s111
                                 + 0.997997 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.35669 w_k1_ADM_s112
                                 + 1.18024 w_k1_AXP_s112
                                 + 1.52653 w_k1_DIS_s112
                                 + 1.16765 w_k1_FDX_s112
                                 + 1.16398 w_k1_IBM_s112
                                 + 1.11339 w_k1_INTC_s112 + 1.4195 w_k1_KO_s112
                                 + 1.13837 w_k1_MAT_s112
                                 + 1.30582 w_k1_MSFT_s112
                                 + 1.19768 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.1058 w_k1_ADM_s112
                                 + 0.827436 w_k1_AXP_s112
                                 + 0.624474 w_k1_DIS_s112
                                 + 0.885206 w_k1_FDX_s112
                                 + 0.63195 w_k1_IBM_s112
                                 + 0.949406 w_k1_INTC_s112
                                 + 0.634146 w_k1_KO_s112
                                 + 0.53218 w_k1_MAT_s112
                                 + 0.801007 w_k1_MSFT_s112
                                 + 0.839762 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.02392 w_k1_ADM_s112
                                 + 1.05004 w_k1_AXP_s112
                                 + 0.889694 w_k1_DIS_s112
                                 + 0.909228 w_k1_FDX_s112
                                 + 1.35418 w_k1_IBM_s112
                                 + 1.02569 w_k1_INTC_s112
                                 + 1.09742 w_k1_KO_s112 + 1.05817 w_k1_MAT_s112
                                 + 0.936979 w_k1_MSFT_s112
                                 + 1.0311 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.1493 w_k1_ADM_s112
                                 + 1.2979 w_k1_AXP_s112 + 1.15012 w_k1_DIS_s112
                                 + 1.28475 w_k1_FDX_s112
                                 + 0.944669 w_k1_IBM_s112
                                 + 1.29495 w_k1_INTC_s112
                                 + 1.01566 w_k1_KO_s112
                                 + 0.657374 w_k1_MAT_s112
                                 + 0.970321 w_k1_MSFT_s112
                                 + 1.01215 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.987622 w_k1_ADM_s112
                                 + 0.975435 w_k1_AXP_s112
                                 + 0.84367 w_k1_DIS_s112
                                 + 0.921952 w_k1_FDX_s112
                                 + 1.05678 w_k1_IBM_s112
                                 + 0.925057 w_k1_INTC_s112
                                 + 1.02738 w_k1_KO_s112 + 1.34053 w_k1_MAT_s112
                                 + 0.913408 w_k1_MSFT_s112 + 1.003 w_k1_PG_s112
                                 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.08469 w_k1_ADM_s112
                                 + 1.31536 w_k1_AXP_s112
                                 + 1.15506 w_k1_DIS_s112
                                 + 1.28441 w_k1_FDX_s112
                                 + 0.855252 w_k1_IBM_s112
                                 + 1.32447 w_k1_INTC_s112
                                 + 0.952389 w_k1_KO_s112
                                 + 0.899141 w_k1_MAT_s112
                                 + 1.11807 w_k1_MSFT_s112
                                 + 1.09131 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.859003 w_k1_ADM_s112
                                 + 0.67061 w_k1_AXP_s112
                                 + 0.985696 w_k1_DIS_s112
                                 + 0.990477 w_k1_FDX_s112
                                 + 0.816074 w_k1_IBM_s112
                                 + 1.08509 w_k1_INTC_s112
                                 + 1.02107 w_k1_KO_s112 + 0.69657 w_k1_MAT_s112
                                 + 1.12305 w_k1_MSFT_s112
                                 + 0.978671 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.1509 w_k1_ADM_s112
                                 + 1.5114 w_k1_AXP_s112 + 1.53624 w_k1_DIS_s112
                                 + 1.24886 w_k1_FDX_s112
                                 + 1.61512 w_k1_IBM_s112
                                 + 1.27893 w_k1_INTC_s112
                                 + 1.40547 w_k1_KO_s112 + 1.68055 w_k1_MAT_s112
                                 + 1.36148 w_k1_MSFT_s112 + 1.2171 w_k1_PG_s112
                                 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.801797 w_k1_ADM_s113
                                 + 0.904505 w_k1_AXP_s113
                                 + 1.37089 w_k1_DIS_s113
                                 + 1.01094 w_k1_FDX_s113 + 1.7104 w_k1_IBM_s113
                                 + 1.00259 w_k1_INTC_s113
                                 + 1.42569 w_k1_KO_s113 + 1.73293 w_k1_MAT_s113
                                 + 1.14192 w_k1_MSFT_s113
                                 + 1.02774 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.801347 w_k1_ADM_s113
                                 + 0.97116 w_k1_AXP_s113
                                 + 1.05892 w_k1_DIS_s113
                                 + 0.912017 w_k1_FDX_s113
                                 + 0.844753 w_k1_IBM_s113
                                 + 1.18343 w_k1_INTC_s113
                                 + 0.979101 w_k1_KO_s113
                                 + 0.763205 w_k1_MAT_s113
                                 + 1.02735 w_k1_MSFT_s113
                                 + 0.874154 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.04546 w_k1_ADM_s113
                                 + 0.807108 w_k1_AXP_s113
                                 + 0.890703 w_k1_DIS_s113
                                 + 0.954136 w_k1_FDX_s113
                                 + 0.70361 w_k1_IBM_s113
                                 + 0.936538 w_k1_INTC_s113
                                 + 0.911335 w_k1_KO_s113
                                 + 0.870263 w_k1_MAT_s113
                                 + 1.02725 w_k1_MSFT_s113
                                 + 0.890634 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.11707 w_k1_ADM_s113
                                 + 1.03982 w_k1_AXP_s113
                                 + 1.14008 w_k1_DIS_s113 + 1.0253 w_k1_FDX_s113
                                 + 1.06734 w_k1_IBM_s113
                                 + 1.24372 w_k1_INTC_s113
                                 + 1.26088 w_k1_KO_s113 + 1.14656 w_k1_MAT_s113
                                 + 1.2381 w_k1_MSFT_s113 + 1.00001 w_k1_PG_s113
                                 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.934635 w_k1_ADM_s113
                                 + 0.720248 w_k1_AXP_s113
                                 + 0.767745 w_k1_DIS_s113
                                 + 0.858554 w_k1_FDX_s113
                                 + 0.55062 w_k1_IBM_s113
                                 + 0.922354 w_k1_INTC_s113
                                 + 0.697863 w_k1_KO_s113
                                 + 0.421174 w_k1_MAT_s113
                                 + 0.962131 w_k1_MSFT_s113
                                 + 0.770847 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.03729 w_k1_ADM_s113
                                 + 0.82221 w_k1_AXP_s113
                                 + 0.804786 w_k1_DIS_s113
                                 + 0.798186 w_k1_FDX_s113
                                 + 0.975785 w_k1_IBM_s113
                                 + 1.03693 w_k1_INTC_s113
                                 + 0.911297 w_k1_KO_s113
                                 + 0.577962 w_k1_MAT_s113
                                 + 0.9292 w_k1_MSFT_s113
                                 + 0.867563 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.22723 w_k1_ADM_s113
                                 + 1.78278 w_k1_AXP_s113
                                 + 1.34375 w_k1_DIS_s113
                                 + 1.23151 w_k1_FDX_s113
                                 + 1.14203 w_k1_IBM_s113
                                 + 1.34864 w_k1_INTC_s113
                                 + 1.30086 w_k1_KO_s113 + 1.38142 w_k1_MAT_s113
                                 + 1.10278 w_k1_MSFT_s113
                                 + 1.11965 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.07611 w_k1_ADM_s113
                                 + 0.90293 w_k1_AXP_s113
                                 + 0.90639 w_k1_DIS_s113
                                 + 0.797608 w_k1_FDX_s113
                                 + 1.22293 w_k1_IBM_s113
                                 + 0.741816 w_k1_INTC_s113
                                 + 1.04555 w_k1_KO_s113 + 1.04624 w_k1_MAT_s113
                                 + 0.869518 w_k1_MSFT_s113
                                 + 0.868856 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.1208 w_k1_ADM_s114
                                 + 1.08626 w_k1_AXP_s114
                                 + 0.94468 w_k1_DIS_s114
                                 + 1.04664 w_k1_FDX_s114 + 1.0162 w_k1_IBM_s114
                                 + 1.00293 w_k1_INTC_s114
                                 + 1.00509 w_k1_KO_s114
                                 + 0.873293 w_k1_MAT_s114
                                 + 1.00874 w_k1_MSFT_s114
                                 + 1.05263 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.920484 w_k1_ADM_s114
                                 + 0.71657 w_k1_AXP_s114
                                 + 0.570447 w_k1_DIS_s114
                                 + 0.754078 w_k1_FDX_s114
                                 + 0.742812 w_k1_IBM_s114
                                 + 0.716305 w_k1_INTC_s114
                                 + 0.708745 w_k1_KO_s114
                                 + 0.553685 w_k1_MAT_s114
                                 + 0.73113 w_k1_MSFT_s114
                                 + 0.782538 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.24627 w_k1_ADM_s114
                                 + 1.04478 w_k1_AXP_s114
                                 + 1.11148 w_k1_DIS_s114
                                 + 0.974262 w_k1_FDX_s114
                                 + 1.27787 w_k1_IBM_s114
                                 + 1.09474 w_k1_INTC_s114
                                 + 1.14767 w_k1_KO_s114 + 1.45291 w_k1_MAT_s114
                                 + 0.930622 w_k1_MSFT_s114
                                 + 0.969099 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.900329 w_k1_ADM_s114
                                 + 0.849405 w_k1_AXP_s114
                                 + 0.834708 w_k1_DIS_s114
                                 + 0.967652 w_k1_FDX_s114
                                 + 0.64126 w_k1_IBM_s114
                                 + 1.00608 w_k1_INTC_s114
                                 + 0.891487 w_k1_KO_s114
                                 + 0.966411 w_k1_MAT_s114
                                 + 0.908006 w_k1_MSFT_s114
                                 + 1.00423 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.6197 w_k1_ADM_s114
                                 + 1.41855 w_k1_AXP_s114
                                 + 1.62588 w_k1_DIS_s114
                                 + 1.28469 w_k1_FDX_s114
                                 + 1.35564 w_k1_IBM_s114
                                 + 1.19155 w_k1_INTC_s114
                                 + 1.37691 w_k1_KO_s114 + 1.44656 w_k1_MAT_s114
                                 + 1.29787 w_k1_MSFT_s114
                                 + 1.18378 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.04878 w_k1_ADM_s114
                                 + 0.940864 w_k1_AXP_s114
                                 + 0.834064 w_k1_DIS_s114
                                 + 0.802801 w_k1_FDX_s114
                                 + 1.23249 w_k1_IBM_s114
                                 + 0.804199 w_k1_INTC_s114
                                 + 0.903982 w_k1_KO_s114
                                 + 1.23694 w_k1_MAT_s114
                                 + 0.82782 w_k1_MSFT_s114
                                 + 0.928011 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.33336 w_k1_ADM_s114
                                 + 0.891839 w_k1_AXP_s114
                                 + 1.05787 w_k1_DIS_s114
                                 + 1.23154 w_k1_FDX_s114
                                 + 0.669784 w_k1_IBM_s114
                                 + 1.0154 w_k1_INTC_s114
                                 + 0.827122 w_k1_KO_s114
                                 + 0.624466 w_k1_MAT_s114
                                 + 1.05227 w_k1_MSFT_s114 + 1.0026 w_k1_PG_s114
                                 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.976531 w_k1_ADM_s114
                                 + 1.02656 w_k1_AXP_s114
                                 + 0.794553 w_k1_DIS_s114
                                 + 0.849937 w_k1_FDX_s114
                                 + 0.955109 w_k1_IBM_s114
                                 + 0.854795 w_k1_INTC_s114
                                 + 1.05385 w_k1_KO_s114 + 1.52135 w_k1_MAT_s114
                                 + 0.954176 w_k1_MSFT_s114
                                 + 0.977755 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.30275 w_k1_ADM_s115
                                 + 1.30352 w_k1_AXP_s115
                                 + 1.27482 w_k1_DIS_s115
                                 + 1.23285 w_k1_FDX_s115
                                 + 0.959163 w_k1_IBM_s115
                                 + 1.198 w_k1_INTC_s115 + 1.01983 w_k1_KO_s115
                                 + 1.01378 w_k1_MAT_s115
                                 + 1.13149 w_k1_MSFT_s115
                                 + 1.04764 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.808768 w_k1_ADM_s115
                                 + 0.863513 w_k1_AXP_s115
                                 + 0.799372 w_k1_DIS_s115
                                 + 0.77465 w_k1_FDX_s115
                                 + 1.05562 w_k1_IBM_s115
                                 + 0.973738 w_k1_INTC_s115
                                 + 0.907019 w_k1_KO_s115
                                 + 1.04286 w_k1_MAT_s115
                                 + 0.947663 w_k1_MSFT_s115
                                 + 0.875806 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.876266 w_k1_ADM_s115
                                 + 1.02826 w_k1_AXP_s115
                                 + 0.93046 w_k1_DIS_s115
                                 + 1.05778 w_k1_FDX_s115
                                 + 0.767763 w_k1_IBM_s115
                                 + 1.02905 w_k1_INTC_s115
                                 + 0.911797 w_k1_KO_s115
                                 + 0.71934 w_k1_MAT_s115
                                 + 1.0453 w_k1_MSFT_s115
                                 + 0.977368 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.02414 w_k1_ADM_s115
                                 + 0.897004 w_k1_AXP_s115
                                 + 0.862507 w_k1_DIS_s115
                                 + 0.937812 w_k1_FDX_s115
                                 + 1.00801 w_k1_IBM_s115
                                 + 0.792261 w_k1_INTC_s115
                                 + 0.949773 w_k1_KO_s115
                                 + 1.07716 w_k1_MAT_s115
                                 + 0.919997 w_k1_MSFT_s115
                                 + 0.867787 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.906267 w_k1_ADM_s115
                                 + 1.20064 w_k1_AXP_s115
                                 + 1.02669 w_k1_DIS_s115
                                 + 0.960339 w_k1_FDX_s115
                                 + 1.47373 w_k1_IBM_s115
                                 + 1.05628 w_k1_INTC_s115
                                 + 1.24627 w_k1_KO_s115 + 1.98101 w_k1_MAT_s115
                                 + 1.07013 w_k1_MSFT_s115
                                 + 1.06556 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.738724 w_k1_ADM_s115
                                 + 0.601728 w_k1_AXP_s115
                                 + 0.629005 w_k1_DIS_s115
                                 + 0.767273 w_k1_FDX_s115
                                 + 0.591613 w_k1_IBM_s115
                                 + 0.764005 w_k1_INTC_s115
                                 + 0.633026 w_k1_KO_s115
                                 + 0.55945 w_k1_MAT_s115
                                 + 0.876307 w_k1_MSFT_s115
                                 + 0.750048 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.75634 w_k1_ADM_s115
                                 + 0.6223 w_k1_AXP_s115 + 0.53623 w_k1_DIS_s115
                                 + 0.677135 w_k1_FDX_s115
                                 + 0.686159 w_k1_IBM_s115
                                 + 0.730004 w_k1_INTC_s115
                                 + 0.621725 w_k1_KO_s115
                                 + 0.782006 w_k1_MAT_s115
                                 + 0.801471 w_k1_MSFT_s115
                                 + 0.780872 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.54438 w_k1_ADM_s115
                                 + 1.35139 w_k1_AXP_s115
                                 + 1.61164 w_k1_DIS_s115
                                 + 1.34754 w_k1_FDX_s115
                                 + 1.21571 w_k1_IBM_s115
                                 + 1.01485 w_k1_INTC_s115
                                 + 1.29817 w_k1_KO_s115 + 1.73083 w_k1_MAT_s115
                                 + 1.30382 w_k1_MSFT_s115
                                 + 1.15038 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.31414 w_k1_ADM_s116
                                 + 1.67305 w_k1_AXP_s116
                                 + 1.68909 w_k1_DIS_s116
                                 + 1.53133 w_k1_FDX_s116
                                 + 1.11238 w_k1_IBM_s116
                                 + 1.55589 w_k1_INTC_s116 + 1.3294 w_k1_KO_s116
                                 + 1.39859 w_k1_MAT_s116
                                 + 1.28829 w_k1_MSFT_s116 + 1.2294 w_k1_PG_s116
                                 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.13163 w_k1_ADM_s116
                                 + 1.04376 w_k1_AXP_s116
                                 + 1.23416 w_k1_DIS_s116
                                 + 1.15035 w_k1_FDX_s116
                                 + 0.905226 w_k1_IBM_s116
                                 + 1.27654 w_k1_INTC_s116
                                 + 1.04813 w_k1_KO_s116
                                 + 0.895924 w_k1_MAT_s116
                                 + 1.10585 w_k1_MSFT_s116
                                 + 0.952996 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.927941 w_k1_ADM_s116
                                 + 1.18628 w_k1_AXP_s116
                                 + 0.857228 w_k1_DIS_s116
                                 + 0.922913 w_k1_FDX_s116
                                 + 1.09554 w_k1_IBM_s116
                                 + 0.993113 w_k1_INTC_s116
                                 + 1.00944 w_k1_KO_s116 + 1.54517 w_k1_MAT_s116
                                 + 0.947498 w_k1_MSFT_s116
                                 + 0.939524 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.915855 w_k1_ADM_s116
                                 + 0.842925 w_k1_AXP_s116
                                 + 0.902835 w_k1_DIS_s116
                                 + 0.887854 w_k1_FDX_s116
                                 + 0.678543 w_k1_IBM_s116
                                 + 1.00175 w_k1_INTC_s116
                                 + 0.94301 w_k1_KO_s116
                                 + 0.607919 w_k1_MAT_s116
                                 + 0.893439 w_k1_MSFT_s116
                                 + 0.862996 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.36174 w_k1_ADM_s116
                                 + 1.53362 w_k1_AXP_s116
                                 + 1.33079 w_k1_DIS_s116 + 1.124 w_k1_FDX_s116
                                 + 1.48522 w_k1_IBM_s116
                                 + 1.15691 w_k1_INTC_s116
                                 + 1.31068 w_k1_KO_s116 + 1.72849 w_k1_MAT_s116
                                 + 0.941507 w_k1_MSFT_s116
                                 + 1.01958 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.17571 w_k1_ADM_s116
                                 + 1.14186 w_k1_AXP_s116
                                 + 1.23891 w_k1_DIS_s116
                                 + 1.06467 w_k1_FDX_s116
                                 + 1.38834 w_k1_IBM_s116
                                 + 1.01184 w_k1_INTC_s116
                                 + 1.15336 w_k1_KO_s116 + 1.28955 w_k1_MAT_s116
                                 + 0.983527 w_k1_MSFT_s116
                                 + 1.11511 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.967828 w_k1_ADM_s116
                                 + 0.885359 w_k1_AXP_s116
                                 + 1.14154 w_k1_DIS_s116
                                 + 0.926906 w_k1_FDX_s116
                                 + 1.17732 w_k1_IBM_s116
                                 + 1.00826 w_k1_INTC_s116
                                 + 1.24326 w_k1_KO_s116 + 1.11391 w_k1_MAT_s116
                                 + 0.930897 w_k1_MSFT_s116
                                 + 0.944691 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.914139 w_k1_ADM_s116
                                 + 0.853177 w_k1_AXP_s116
                                 + 0.764076 w_k1_DIS_s116
                                 + 0.808561 w_k1_FDX_s116
                                 + 0.907791 w_k1_IBM_s116
                                 + 0.798158 w_k1_INTC_s116
                                 + 0.85603 w_k1_KO_s116 + 1.11909 w_k1_MAT_s116
                                 + 0.770827 w_k1_MSFT_s116
                                 + 0.885899 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.748353 w_k1_ADM_s117
                                 + 0.840958 w_k1_AXP_s117
                                 + 0.686463 w_k1_DIS_s117
                                 + 0.781151 w_k1_FDX_s117
                                 + 0.94764 w_k1_IBM_s117
                                 + 0.834849 w_k1_INTC_s117
                                 + 0.717432 w_k1_KO_s117
                                 + 0.807236 w_k1_MAT_s117
                                 + 0.799954 w_k1_MSFT_s117
                                 + 0.828189 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.92491 w_k1_ADM_s117
                                 + 0.788283 w_k1_AXP_s117
                                 + 0.809996 w_k1_DIS_s117
                                 + 0.924059 w_k1_FDX_s117
                                 + 0.690146 w_k1_IBM_s117
                                 + 0.912863 w_k1_INTC_s117
                                 + 0.703735 w_k1_KO_s117
                                 + 0.463253 w_k1_MAT_s117
                                 + 0.842002 w_k1_MSFT_s117
                                 + 0.831465 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.881867 w_k1_ADM_s117
                                 + 0.722885 w_k1_AXP_s117
                                 + 0.825892 w_k1_DIS_s117
                                 + 0.80265 w_k1_FDX_s117
                                 + 1.23287 w_k1_IBM_s117
                                 + 0.939632 w_k1_INTC_s117
                                 + 0.993487 w_k1_KO_s117
                                 + 0.558938 w_k1_MAT_s117
                                 + 0.877709 w_k1_MSFT_s117
                                 + 0.960699 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.941612 w_k1_ADM_s117
                                 + 0.794131 w_k1_AXP_s117
                                 + 0.953157 w_k1_DIS_s117
                                 + 0.97803 w_k1_FDX_s117
                                 + 0.868944 w_k1_IBM_s117
                                 + 0.956263 w_k1_INTC_s117
                                 + 0.913625 w_k1_KO_s117
                                 + 0.652393 w_k1_MAT_s117
                                 + 0.935171 w_k1_MSFT_s117
                                 + 1.04641 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.29259 w_k1_ADM_s117
                                 + 1.12498 w_k1_AXP_s117
                                 + 1.64644 w_k1_DIS_s117
                                 + 1.32491 w_k1_FDX_s117
                                 + 1.19262 w_k1_IBM_s117
                                 + 1.12549 w_k1_INTC_s117
                                 + 1.33428 w_k1_KO_s117 + 1.19489 w_k1_MAT_s117
                                 + 1.23513 w_k1_MSFT_s117
                                 + 1.25096 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.04011 w_k1_ADM_s117
                                 + 0.939414 w_k1_AXP_s117
                                 + 1.16263 w_k1_DIS_s117
                                 + 1.00743 w_k1_FDX_s117 + 1.1063 w_k1_IBM_s117
                                 + 0.895256 w_k1_INTC_s117
                                 + 1.12469 w_k1_KO_s117
                                 + 0.999233 w_k1_MAT_s117
                                 + 1.08276 w_k1_MSFT_s117
                                 + 1.07827 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.22268 w_k1_ADM_s117
                                 + 1.1743 w_k1_AXP_s117 + 1.42825 w_k1_DIS_s117
                                 + 1.08886 w_k1_FDX_s117
                                 + 1.60258 w_k1_IBM_s117
                                 + 1.18492 w_k1_INTC_s117
                                 + 1.34638 w_k1_KO_s117 + 1.23424 w_k1_MAT_s117
                                 + 0.989811 w_k1_MSFT_s117
                                 + 1.13356 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.07883 w_k1_ADM_s117
                                 + 1.14402 w_k1_AXP_s117
                                 + 1.29442 w_k1_DIS_s117
                                 + 1.14035 w_k1_FDX_s117
                                 + 0.892669 w_k1_IBM_s117
                                 + 1.09337 w_k1_INTC_s117
                                 + 1.00611 w_k1_KO_s117
                                 + 0.618037 w_k1_MAT_s117
                                 + 1.10977 w_k1_MSFT_s117
                                 + 1.09844 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0205671886147459 w_k1_ADM_s000
                                 + 0.033252363136457 w_k1_ADM_s001
                                 + 0.0252656568359809 w_k1_ADM_s002
                                 + 0.0163382371168401 w_k1_ADM_s003
                                 + 0.01588638940568 w_k1_ADM_s004
                                 + 0.0243547394695978 w_k1_ADM_s005
                                 + 0.0281850804515725 w_k1_ADM_s006
                                 + 0.00661061829605955 w_k1_ADM_s007
                                 + 0.0183875854374124 w_k1_ADM_s010
                                 + 0.0229680174609586 w_k1_ADM_s011
                                 + 0.0147209865817684 w_k1_ADM_s012
                                 + 0.0150785417205668 w_k1_ADM_s013
                                 + 0.0226042845261175 w_k1_ADM_s014
                                 + 0.0149928601255015 w_k1_ADM_s015
                                 + 0.0235031961280048 w_k1_ADM_s016
                                 + 0.0245793122470392 w_k1_ADM_s017
                                 + 0.0295719244690462 w_k1_ADM_s100
                                 + 0.0512707288990508 w_k1_ADM_s101
                                 + 0.056975572278498 w_k1_ADM_s102
                                 + 0.03597155995501 w_k1_ADM_s103
                                 + 0.0532678245040292 w_k1_ADM_s104
                                 + 0.0615647731500158 w_k1_ADM_s105
                                 + 0.0355444838057216 w_k1_ADM_s106
                                 + 0.0995518529865841 w_k1_ADM_s107
                                 + 0.0223408718292113 w_k1_ADM_s110
                                 + 0.0220448993755725 w_k1_ADM_s111
                                 + 0.0357077806373221 w_k1_ADM_s112
                                 + 0.0217364778543947 w_k1_ADM_s113
                                 + 0.0234948691812537 w_k1_ADM_s114
                                 + 0.0478625337928051 w_k1_ADM_s115
                                 + 0.0457474687288239 w_k1_ADM_s116
                                 + 0.0532764047725174 w_k1_ADM_s117
                                 + 0.0248874560382929 w_k1_AXP_s000
                                 + 0.0320611782542061 w_k1_AXP_s001
                                 + 0.021847296224372 w_k1_AXP_s002
                                 + 0.023478485581179 w_k1_AXP_s003
                                 + 0.0146245171553038 w_k1_AXP_s004
                                 + 0.0234432959347103 w_k1_AXP_s005
                                 + 0.0275528921015117 w_k1_AXP_s006
                                 + 0.00650602822436205 w_k1_AXP_s007
                                 + 0.0176192189655863 w_k1_AXP_s010
                                 + 0.0253997948736571 w_k1_AXP_s011
                                 + 0.0152216836574435 w_k1_AXP_s012
                                 + 0.0155661477314581 w_k1_AXP_s013
                                 + 0.0276960975163254 w_k1_AXP_s014
                                 + 0.0155129460232565 w_k1_AXP_s015
                                 + 0.0225368156188294 w_k1_AXP_s016
                                 + 0.0202497193636995 w_k1_AXP_s017
                                 + 0.0253796452484621 w_k1_AXP_s100
                                 + 0.0552125631834392 w_k1_AXP_s101
                                 + 0.0521778071912384 w_k1_AXP_s102
                                 + 0.0366812851486602 w_k1_AXP_s103
                                 + 0.0603589660942355 w_k1_AXP_s104
                                 + 0.0595121146716578 w_k1_AXP_s105
                                 + 0.0409671620732157 w_k1_AXP_s106
                                 + 0.102421382277511 w_k1_AXP_s107
                                 + 0.0212454510887069 w_k1_AXP_s110
                                 + 0.0214653183425075 w_k1_AXP_s111
                                 + 0.0342568271092973 w_k1_AXP_s112
                                 + 0.022934941279177 w_k1_AXP_s113
                                 + 0.0206283559872432 w_k1_AXP_s114
                                 + 0.048745475854665 w_k1_AXP_s115
                                 + 0.0466070547387217 w_k1_AXP_s116
                                 + 0.0494189255195761 w_k1_AXP_s117
                                 + 0.0230488260163318 w_k1_DIS_s000
                                 + 0.0326081134834167 w_k1_DIS_s001
                                 + 0.0218767326753229 w_k1_DIS_s002
                                 + 0.0209858756630802 w_k1_DIS_s003
                                 + 0.0144390381141493 w_k1_DIS_s004
                                 + 0.0248921046650035 w_k1_DIS_s005
                                 + 0.0297922029495312 w_k1_DIS_s006
                                 + 0.00679917322730829 w_k1_DIS_s007
                                 + 0.017434657751794 w_k1_DIS_s010
                                 + 0.02451712091171 w_k1_DIS_s011
                                 + 0.0167962274954911 w_k1_DIS_s012
                                 + 0.0141840469836681 w_k1_DIS_s013
                                 + 0.0266271623399469 w_k1_DIS_s014
                                 + 0.0151003425165095 w_k1_DIS_s015
                                 + 0.0232491595234009 w_k1_DIS_s016
                                 + 0.0213333031057831 w_k1_DIS_s017
                                 + 0.027604873563661 w_k1_DIS_s100
                                 + 0.0608558103490324 w_k1_DIS_s101
                                 + 0.053523358334393 w_k1_DIS_s102
                                 + 0.0366721896079544 w_k1_DIS_s103
                                 + 0.0580135332121638 w_k1_DIS_s104
                                 + 0.0616390445974761 w_k1_DIS_s105
                                 + 0.0389538646145426 w_k1_DIS_s106
                                 + 0.0934830185316074 w_k1_DIS_s107
                                 + 0.0235202709686805 w_k1_DIS_s110
                                 + 0.0213522892761836 w_k1_DIS_s111
                                 + 0.0345859473277254 w_k1_DIS_s112
                                 + 0.022827219097341 w_k1_DIS_s113
                                 + 0.02034861295706 w_k1_DIS_s114
                                 + 0.0472328074569271 w_k1_DIS_s115
                                 + 0.0466370817249854 w_k1_DIS_s116
                                 + 0.0575276474446655 w_k1_DIS_s117
                                 + 0.0216365586887963 w_k1_FDX_s000
                                 + 0.031660454266259 w_k1_FDX_s001
                                 + 0.0221821573656909 w_k1_FDX_s002
                                 + 0.0222493443346371 w_k1_FDX_s003
                                 + 0.0146833840970939 w_k1_FDX_s004
                                 + 0.0231376832877567 w_k1_FDX_s005
                                 + 0.0290201530255756 w_k1_FDX_s006
                                 + 0.00645587517836593 w_k1_FDX_s007
                                 + 0.01808082317754 w_k1_FDX_s010
                                 + 0.0238342978415037 w_k1_FDX_s011
                                 + 0.0159207017108805 w_k1_FDX_s012
                                 + 0.0142138968260521 w_k1_FDX_s013
                                 + 0.0262435747187265 w_k1_FDX_s014
                                 + 0.014269277284451 w_k1_FDX_s015
                                 + 0.0224086735295329 w_k1_FDX_s016
                                 + 0.0213185869401854 w_k1_FDX_s017
                                 + 0.0263725341038047 w_k1_FDX_s100
                                 + 0.0532730980668952 w_k1_FDX_s101
                                 + 0.0549951919846105 w_k1_FDX_s102
                                 + 0.034045108878643 w_k1_FDX_s103
                                 + 0.0580969769196722 w_k1_FDX_s104
                                 + 0.0607474773757471 w_k1_FDX_s105
                                 + 0.0402206156079202 w_k1_FDX_s106
                                 + 0.0945070105633332 w_k1_FDX_s107
                                 + 0.0230144767099534 w_k1_FDX_s110
                                 + 0.0222352705719253 w_k1_FDX_s111
                                 + 0.03501315551466 w_k1_FDX_s112
                                 + 0.0208929465049049 w_k1_FDX_s113
                                 + 0.0197826466646088 w_k1_FDX_s114
                                 + 0.0471667129122592 w_k1_FDX_s115
                                 + 0.0436593669114746 w_k1_FDX_s116
                                 + 0.0525859446351243 w_k1_FDX_s117
                                 + 0.0221593496631348 w_k1_IBM_s000
                                 + 0.032747472663767 w_k1_IBM_s001
                                 + 0.0197833667827679 w_k1_IBM_s002
                                 + 0.021503624757775 w_k1_IBM_s003
                                 + 0.0148885573835408 w_k1_IBM_s004
                                 + 0.023174810511209 w_k1_IBM_s005
                                 + 0.0283437075741185 w_k1_IBM_s006
                                 + 0.00622223453592556 w_k1_IBM_s007
                                 + 0.0186719411439405 w_k1_IBM_s010
                                 + 0.024417695322216 w_k1_IBM_s011
                                 + 0.0164244917740498 w_k1_IBM_s012
                                 + 0.0156052700512773 w_k1_IBM_s013
                                 + 0.0257958635200788 w_k1_IBM_s014
                                 + 0.0144176165295552 w_k1_IBM_s015
                                 + 0.0213677192278392 w_k1_IBM_s016
                                 + 0.0223717892410983 w_k1_IBM_s017
                                 + 0.0261229357984136 w_k1_IBM_s100
                                 + 0.0566951036430761 w_k1_IBM_s101
                                 + 0.0534903524920621 w_k1_IBM_s102
                                 + 0.035303103494207 w_k1_IBM_s103
                                 + 0.059502964687667 w_k1_IBM_s104
                                 + 0.0635374933127708 w_k1_IBM_s105
                                 + 0.0402843770829988 w_k1_IBM_s106
                                 + 0.0951294734593726 w_k1_IBM_s107
                                 + 0.0234600870887163 w_k1_IBM_s110
                                 + 0.0217071953464641 w_k1_IBM_s111
                                 + 0.0350143602976131 w_k1_IBM_s112
                                 + 0.0217478113317712 w_k1_IBM_s113
                                 + 0.0215383656573758 w_k1_IBM_s114
                                 + 0.0492110109672546 w_k1_IBM_s115
                                 + 0.0457311162321047 w_k1_IBM_s116
                                 + 0.0554965363442392 w_k1_IBM_s117
                                 + 0.0267260090397364 w_k1_INTC_s000
                                 + 0.0354850694101939 w_k1_INTC_s001
                                 + 0.0231670228701876 w_k1_INTC_s002
                                 + 0.0226193303147965 w_k1_INTC_s003
                                 + 0.0169368323526971 w_k1_INTC_s004
                                 + 0.025580519275104 w_k1_INTC_s005
                                 + 0.0340298439679661 w_k1_INTC_s006
                                 + 0.00546335664810118 w_k1_INTC_s007
                                 + 0.0166169301217859 w_k1_INTC_s010
                                 + 0.0196253984212094 w_k1_INTC_s011
                                 + 0.0152106089520289 w_k1_INTC_s012
                                 + 0.0134060271334491 w_k1_INTC_s013
                                 + 0.0261762379835727 w_k1_INTC_s014
                                 + 0.0142404592030269 w_k1_INTC_s015
                                 + 0.0227348005133493 w_k1_INTC_s016
                                 + 0.0218462965644809 w_k1_INTC_s017
                                 + 0.0252568299711454 w_k1_INTC_s100
                                 + 0.0520617296033502 w_k1_INTC_s101
                                 + 0.0575010030610604 w_k1_INTC_s102
                                 + 0.0344326570091201 w_k1_INTC_s103
                                 + 0.05763540034951 w_k1_INTC_s104
                                 + 0.0537411781536119 w_k1_INTC_s105
                                 + 0.0455705880720412 w_k1_INTC_s106
                                 + 0.0968348752188405 w_k1_INTC_s107
                                 + 0.0255218798286281 w_k1_INTC_s110
                                 + 0.0220833007205121 w_k1_INTC_s111
                                 + 0.0368512578554052 w_k1_INTC_s112
                                 + 0.0234960859634629 w_k1_INTC_s113
                                 + 0.019510361052459 w_k1_INTC_s114
                                 + 0.0458077556546084 w_k1_INTC_s115
                                 + 0.0452472248993955 w_k1_INTC_s116
                                 + 0.0520687095530327 w_k1_INTC_s117
                                 + 0.0229188288615137 w_k1_KO_s000
                                 + 0.0337825781317566 w_k1_KO_s001
                                 + 0.0209515689156267 w_k1_KO_s002
                                 + 0.0194242530640984 w_k1_KO_s003
                                 + 0.014615819882733 w_k1_KO_s004
                                 + 0.0222524048193244 w_k1_KO_s005
                                 + 0.0311518768035624 w_k1_KO_s006
                                 + 0.00680278021303524 w_k1_KO_s007
                                 + 0.0161772006336788 w_k1_KO_s010
                                 + 0.0241702740335441 w_k1_KO_s011
                                 + 0.0143926657373029 w_k1_KO_s012
                                 + 0.0138203529258216 w_k1_KO_s013
                                 + 0.0234952443161176 w_k1_KO_s014
                                 + 0.0147551031368765 w_k1_KO_s015
                                 + 0.0206207773997813 w_k1_KO_s016
                                 + 0.0208060090396428 w_k1_KO_s017
                                 + 0.0257335599623276 w_k1_KO_s100
                                 + 0.0594425221874502 w_k1_KO_s101
                                 + 0.0486836384879197 w_k1_KO_s102
                                 + 0.0341648440857389 w_k1_KO_s103
                                 + 0.0565015791026767 w_k1_KO_s104
                                 + 0.0609435748416798 w_k1_KO_s105
                                 + 0.0403173014114652 w_k1_KO_s106
                                 + 0.0894676335426846 w_k1_KO_s107
                                 + 0.0227616686156549 w_k1_KO_s110
                                 + 0.0234670548350578 w_k1_KO_s111
                                 + 0.0354540686271364 w_k1_KO_s112
                                 + 0.0231852841888955 w_k1_KO_s113
                                 + 0.020446122330377 w_k1_KO_s114
                                 + 0.0477003538353308 w_k1_KO_s115
                                 + 0.0459463801902162 w_k1_KO_s116
                                 + 0.0530474865430599 w_k1_KO_s117
                                 + 0.0231876803700699 w_k1_MAT_s000
                                 + 0.0344092004807603 w_k1_MAT_s001
                                 + 0.0183354839667269 w_k1_MAT_s002
                                 + 0.0205506123185915 w_k1_MAT_s003
                                 + 0.0155424595094075 w_k1_MAT_s004
                                 + 0.0200750327025103 w_k1_MAT_s005
                                 + 0.0222022927187398 w_k1_MAT_s006
                                 + 0.010592992546378 w_k1_MAT_s007
                                 + 0.0146293917986818 w_k1_MAT_s010
                                 + 0.0268936801579148 w_k1_MAT_s011
                                 + 0.018877691257984 w_k1_MAT_s012
                                 + 0.0146454526559552 w_k1_MAT_s013
                                 + 0.0237258630152587 w_k1_MAT_s014
                                 + 0.0168016684073423 w_k1_MAT_s015
                                 + 0.0170914666679664 w_k1_MAT_s016
                                 + 0.0206909056072612 w_k1_MAT_s017
                                 + 0.0249739806366789 w_k1_MAT_s100
                                 + 0.0567191606138127 w_k1_MAT_s101
                                 + 0.0466603385477835 w_k1_MAT_s102
                                 + 0.0382507990574889 w_k1_MAT_s103
                                 + 0.0620190232759413 w_k1_MAT_s104
                                 + 0.0579015024342897 w_k1_MAT_s105
                                 + 0.0388494129705515 w_k1_MAT_s106
                                 + 0.112894771029105 w_k1_MAT_s107
                                 + 0.0177268236574947 w_k1_MAT_s110
                                 + 0.0201785285163841 w_k1_MAT_s111
                                 + 0.0323578342221449 w_k1_MAT_s112
                                 + 0.0216394371278906 w_k1_MAT_s113
                                 + 0.0227263552084493 w_k1_MAT_s114
                                 + 0.0566305565985622 w_k1_MAT_s115
                                 + 0.0506220262028948 w_k1_MAT_s116
                                 + 0.0418527312548474 w_k1_MAT_s117
                                 + 0.0230681911150251 w_k1_MSFT_s000
                                 + 0.0316919467992981 w_k1_MSFT_s001
                                 + 0.0221264384215566 w_k1_MSFT_s002
                                 + 0.0216704144870132 w_k1_MSFT_s003
                                 + 0.0153568445531145 w_k1_MSFT_s004
                                 + 0.0208339104272387 w_k1_MSFT_s005
                                 + 0.0319942767305073 w_k1_MSFT_s006
                                 + 0.00646036834691312 w_k1_MSFT_s007
                                 + 0.0186010135539226 w_k1_MSFT_s010
                                 + 0.0223670728482496 w_k1_MSFT_s011
                                 + 0.0153288993538128 w_k1_MSFT_s012
                                 + 0.0134061752672204 w_k1_MSFT_s013
                                 + 0.0240884895394245 w_k1_MSFT_s014
                                 + 0.0148520780797593 w_k1_MSFT_s015
                                 + 0.0225241918820352 w_k1_MSFT_s016
                                 + 0.0225895279691836 w_k1_MSFT_s017
                                 + 0.0242348988595285 w_k1_MSFT_s100
                                 + 0.0530339227655823 w_k1_MSFT_s101
                                 + 0.0550066471645434 w_k1_MSFT_s102
                                 + 0.0341915715747791 w_k1_MSFT_s103
                                 + 0.0564188811231891 w_k1_MSFT_s104
                                 + 0.0596500377640041 w_k1_MSFT_s105
                                 + 0.0432698378425813 w_k1_MSFT_s106
                                 + 0.0904424167450023 w_k1_MSFT_s107
                                 + 0.0234848191881608 w_k1_MSFT_s110
                                 + 0.020571479570279 w_k1_MSFT_s111
                                 + 0.0354103450981946 w_k1_MSFT_s112
                                 + 0.0223709655768738 w_k1_MSFT_s113
                                 + 0.0193219822278689 w_k1_MSFT_s114
                                 + 0.0490355777870145 w_k1_MSFT_s115
                                 + 0.0410705691693637 w_k1_MSFT_s116
                                 + 0.0514151708774128 w_k1_MSFT_s117
                                 + 0.0199835627924811 w_k1_PG_s000
                                 + 0.0293233828572958 w_k1_PG_s001
                                 + 0.0210990944552712 w_k1_PG_s002
                                 + 0.0205470121387834 w_k1_PG_s003
                                 + 0.014231186360789 w_k1_PG_s004
                                 + 0.0240125870442649 w_k1_PG_s005
                                 + 0.0295474386406385 w_k1_PG_s006
                                 + 0.00659882521668792 w_k1_PG_s007
                                 + 0.0163228851600917 w_k1_PG_s010
                                 + 0.0242217217021344 w_k1_PG_s011
                                 + 0.0150624544471669 w_k1_PG_s012
                                 + 0.0146093708710161 w_k1_PG_s013
                                 + 0.0244784365439632 w_k1_PG_s014
                                 + 0.0141045476520727 w_k1_PG_s015
                                 + 0.0219257272318198 w_k1_PG_s016
                                 + 0.0210079483983816 w_k1_PG_s017
                                 + 0.0259498356931852 w_k1_PG_s100
                                 + 0.0586246547904673 w_k1_PG_s101
                                 + 0.0557089049707867 w_k1_PG_s102
                                 + 0.0371427928609647 w_k1_PG_s103
                                 + 0.0585204127619094 w_k1_PG_s104
                                 + 0.0577604536442699 w_k1_PG_s105
                                 + 0.0398457396153741 w_k1_PG_s106
                                 + 0.0949548316041222 w_k1_PG_s107
                                 + 0.0206081168553797 w_k1_PG_s110
                                 + 0.0194272695374075 w_k1_PG_s111
                                 + 0.0347072614666492 w_k1_PG_s112
                                 + 0.0202276779878291 w_k1_PG_s113
                                 + 0.0197913012145185 w_k1_PG_s114
                                 + 0.0457108202282576 w_k1_PG_s115
                                 + 0.0423388251626882 w_k1_PG_s116
                                 + 0.0536562622932253 w_k1_PG_s117 >= 1.18
Bounds
 0 <= w_0_ADM <= 1
 0 <= w_0_AXP <= 1
 0 <= w_0_DIS <= 1
 0 <= w_0_FDX <= 1
 0 <= w_0_IBM <= 1
 0 <= w_0_INTC <= 1
 0 <= w_0_KO <= 1
 0 <= w_0_MAT <= 1
 0 <= w_0_MSFT <= 1
 0 <= w_0_PG <= 1
 0 <= w_k1_ADM_s0 <= 1
 0 <= w_k1_ADM_s1 <= 1
 0 <= w_k1_ADM_s00 <= 1
 0 <= w_k1_ADM_s01 <= 1
 0 <= w_k1_ADM_s10 <= 1
 0 <= w_k1_ADM_s11 <= 1
 0 <= w_k1_ADM_s000 <= 1
 0 <= w_k1_ADM_s001 <= 1
 0 <= w_k1_ADM_s002 <= 1
 0 <= w_k1_ADM_s003 <= 1
 0 <= w_k1_ADM_s004 <= 1
 0 <= w_k1_ADM_s005 <= 1
 0 <= w_k1_ADM_s006 <= 1
 0 <= w_k1_ADM_s007 <= 1
 0 <= w_k1_ADM_s010 <= 1
 0 <= w_k1_ADM_s011 <= 1
 0 <= w_k1_ADM_s012 <= 1
 0 <= w_k1_ADM_s013 <= 1
 0 <= w_k1_ADM_s014 <= 1
 0 <= w_k1_ADM_s015 <= 1
 0 <= w_k1_ADM_s016 <= 1
 0 <= w_k1_ADM_s017 <= 1
 0 <= w_k1_ADM_s100 <= 1
 0 <= w_k1_ADM_s101 <= 1
 0 <= w_k1_ADM_s102 <= 1
 0 <= w_k1_ADM_s103 <= 1
 0 <= w_k1_ADM_s104 <= 1
 0 <= w_k1_ADM_s105 <= 1
 0 <= w_k1_ADM_s106 <= 1
 0 <= w_k1_ADM_s107 <= 1
 0 <= w_k1_ADM_s110 <= 1
 0 <= w_k1_ADM_s111 <= 1
 0 <= w_k1_ADM_s112 <= 1
 0 <= w_k1_ADM_s113 <= 1
 0 <= w_k1_ADM_s114 <= 1
 0 <= w_k1_ADM_s115 <= 1
 0 <= w_k1_ADM_s116 <= 1
 0 <= w_k1_ADM_s117 <= 1
 0 <= w_k1_AXP_s0 <= 1
 0 <= w_k1_AXP_s1 <= 1
 0 <= w_k1_AXP_s00 <= 1
 0 <= w_k1_AXP_s01 <= 1
 0 <= w_k1_AXP_s10 <= 1
 0 <= w_k1_AXP_s11 <= 1
 0 <= w_k1_AXP_s000 <= 1
 0 <= w_k1_AXP_s001 <= 1
 0 <= w_k1_AXP_s002 <= 1
 0 <= w_k1_AXP_s003 <= 1
 0 <= w_k1_AXP_s004 <= 1
 0 <= w_k1_AXP_s005 <= 1
 0 <= w_k1_AXP_s006 <= 1
 0 <= w_k1_AXP_s007 <= 1
 0 <= w_k1_AXP_s010 <= 1
 0 <= w_k1_AXP_s011 <= 1
 0 <= w_k1_AXP_s012 <= 1
 0 <= w_k1_AXP_s013 <= 1
 0 <= w_k1_AXP_s014 <= 1
 0 <= w_k1_AXP_s015 <= 1
 0 <= w_k1_AXP_s016 <= 1
 0 <= w_k1_AXP_s017 <= 1
 0 <= w_k1_AXP_s100 <= 1
 0 <= w_k1_AXP_s101 <= 1
 0 <= w_k1_AXP_s102 <= 1
 0 <= w_k1_AXP_s103 <= 1
 0 <= w_k1_AXP_s104 <= 1
 0 <= w_k1_AXP_s105 <= 1
 0 <= w_k1_AXP_s106 <= 1
 0 <= w_k1_AXP_s107 <= 1
 0 <= w_k1_AXP_s110 <= 1
 0 <= w_k1_AXP_s111 <= 1
 0 <= w_k1_AXP_s112 <= 1
 0 <= w_k1_AXP_s113 <= 1
 0 <= w_k1_AXP_s114 <= 1
 0 <= w_k1_AXP_s115 <= 1
 0 <= w_k1_AXP_s116 <= 1
 0 <= w_k1_AXP_s117 <= 1
 0 <= w_k1_DIS_s0 <= 1
 0 <= w_k1_DIS_s1 <= 1
 0 <= w_k1_DIS_s00 <= 1
 0 <= w_k1_DIS_s01 <= 1
 0 <= w_k1_DIS_s10 <= 1
 0 <= w_k1_DIS_s11 <= 1
 0 <= w_k1_DIS_s000 <= 1
 0 <= w_k1_DIS_s001 <= 1
 0 <= w_k1_DIS_s002 <= 1
 0 <= w_k1_DIS_s003 <= 1
 0 <= w_k1_DIS_s004 <= 1
 0 <= w_k1_DIS_s005 <= 1
 0 <= w_k1_DIS_s006 <= 1
 0 <= w_k1_DIS_s007 <= 1
 0 <= w_k1_DIS_s010 <= 1
 0 <= w_k1_DIS_s011 <= 1
 0 <= w_k1_DIS_s012 <= 1
 0 <= w_k1_DIS_s013 <= 1
 0 <= w_k1_DIS_s014 <= 1
 0 <= w_k1_DIS_s015 <= 1
 0 <= w_k1_DIS_s016 <= 1
 0 <= w_k1_DIS_s017 <= 1
 0 <= w_k1_DIS_s100 <= 1
 0 <= w_k1_DIS_s101 <= 1
 0 <= w_k1_DIS_s102 <= 1
 0 <= w_k1_DIS_s103 <= 1
 0 <= w_k1_DIS_s104 <= 1
 0 <= w_k1_DIS_s105 <= 1
 0 <= w_k1_DIS_s106 <= 1
 0 <= w_k1_DIS_s107 <= 1
 0 <= w_k1_DIS_s110 <= 1
 0 <= w_k1_DIS_s111 <= 1
 0 <= w_k1_DIS_s112 <= 1
 0 <= w_k1_DIS_s113 <= 1
 0 <= w_k1_DIS_s114 <= 1
 0 <= w_k1_DIS_s115 <= 1
 0 <= w_k1_DIS_s116 <= 1
 0 <= w_k1_DIS_s117 <= 1
 0 <= w_k1_FDX_s0 <= 1
 0 <= w_k1_FDX_s1 <= 1
 0 <= w_k1_FDX_s00 <= 1
 0 <= w_k1_FDX_s01 <= 1
 0 <= w_k1_FDX_s10 <= 1
 0 <= w_k1_FDX_s11 <= 1
 0 <= w_k1_FDX_s000 <= 1
 0 <= w_k1_FDX_s001 <= 1
 0 <= w_k1_FDX_s002 <= 1
 0 <= w_k1_FDX_s003 <= 1
 0 <= w_k1_FDX_s004 <= 1
 0 <= w_k1_FDX_s005 <= 1
 0 <= w_k1_FDX_s006 <= 1
 0 <= w_k1_FDX_s007 <= 1
 0 <= w_k1_FDX_s010 <= 1
 0 <= w_k1_FDX_s011 <= 1
 0 <= w_k1_FDX_s012 <= 1
 0 <= w_k1_FDX_s013 <= 1
 0 <= w_k1_FDX_s014 <= 1
 0 <= w_k1_FDX_s015 <= 1
 0 <= w_k1_FDX_s016 <= 1
 0 <= w_k1_FDX_s017 <= 1
 0 <= w_k1_FDX_s100 <= 1
 0 <= w_k1_FDX_s101 <= 1
 0 <= w_k1_FDX_s102 <= 1
 0 <= w_k1_FDX_s103 <= 1
 0 <= w_k1_FDX_s104 <= 1
 0 <= w_k1_FDX_s105 <= 1
 0 <= w_k1_FDX_s106 <= 1
 0 <= w_k1_FDX_s107 <= 1
 0 <= w_k1_FDX_s110 <= 1
 0 <= w_k1_FDX_s111 <= 1
 0 <= w_k1_FDX_s112 <= 1
 0 <= w_k1_FDX_s113 <= 1
 0 <= w_k1_FDX_s114 <= 1
 0 <= w_k1_FDX_s115 <= 1
 0 <= w_k1_FDX_s116 <= 1
 0 <= w_k1_FDX_s117 <= 1
 0 <= w_k1_IBM_s0 <= 1
 0 <= w_k1_IBM_s1 <= 1
 0 <= w_k1_IBM_s00 <= 1
 0 <= w_k1_IBM_s01 <= 1
 0 <= w_k1_IBM_s10 <= 1
 0 <= w_k1_IBM_s11 <= 1
 0 <= w_k1_IBM_s000 <= 1
 0 <= w_k1_IBM_s001 <= 1
 0 <= w_k1_IBM_s002 <= 1
 0 <= w_k1_IBM_s003 <= 1
 0 <= w_k1_IBM_s004 <= 1
 0 <= w_k1_IBM_s005 <= 1
 0 <= w_k1_IBM_s006 <= 1
 0 <= w_k1_IBM_s007 <= 1
 0 <= w_k1_IBM_s010 <= 1
 0 <= w_k1_IBM_s011 <= 1
 0 <= w_k1_IBM_s012 <= 1
 0 <= w_k1_IBM_s013 <= 1
 0 <= w_k1_IBM_s014 <= 1
 0 <= w_k1_IBM_s015 <= 1
 0 <= w_k1_IBM_s016 <= 1
 0 <= w_k1_IBM_s017 <= 1
 0 <= w_k1_IBM_s100 <= 1
 0 <= w_k1_IBM_s101 <= 1
 0 <= w_k1_IBM_s102 <= 1
 0 <= w_k1_IBM_s103 <= 1
 0 <= w_k1_IBM_s104 <= 1
 0 <= w_k1_IBM_s105 <= 1
 0 <= w_k1_IBM_s106 <= 1
 0 <= w_k1_IBM_s107 <= 1
 0 <= w_k1_IBM_s110 <= 1
 0 <= w_k1_IBM_s111 <= 1
 0 <= w_k1_IBM_s112 <= 1
 0 <= w_k1_IBM_s113 <= 1
 0 <= w_k1_IBM_s114 <= 1
 0 <= w_k1_IBM_s115 <= 1
 0 <= w_k1_IBM_s116 <= 1
 0 <= w_k1_IBM_s117 <= 1
 0 <= w_k1_INTC_s0 <= 1
 0 <= w_k1_INTC_s1 <= 1
 0 <= w_k1_INTC_s00 <= 1
 0 <= w_k1_INTC_s01 <= 1
 0 <= w_k1_INTC_s10 <= 1
 0 <= w_k1_INTC_s11 <= 1
 0 <= w_k1_INTC_s000 <= 1
 0 <= w_k1_INTC_s001 <= 1
 0 <= w_k1_INTC_s002 <= 1
 0 <= w_k1_INTC_s003 <= 1
 0 <= w_k1_INTC_s004 <= 1
 0 <= w_k1_INTC_s005 <= 1
 0 <= w_k1_INTC_s006 <= 1
 0 <= w_k1_INTC_s007 <= 1
 0 <= w_k1_INTC_s010 <= 1
 0 <= w_k1_INTC_s011 <= 1
 0 <= w_k1_INTC_s012 <= 1
 0 <= w_k1_INTC_s013 <= 1
 0 <= w_k1_INTC_s014 <= 1
 0 <= w_k1_INTC_s015 <= 1
 0 <= w_k1_INTC_s016 <= 1
 0 <= w_k1_INTC_s017 <= 1
 0 <= w_k1_INTC_s100 <= 1
 0 <= w_k1_INTC_s101 <= 1
 0 <= w_k1_INTC_s102 <= 1
 0 <= w_k1_INTC_s103 <= 1
 0 <= w_k1_INTC_s104 <= 1
 0 <= w_k1_INTC_s105 <= 1
 0 <= w_k1_INTC_s106 <= 1
 0 <= w_k1_INTC_s107 <= 1
 0 <= w_k1_INTC_s110 <= 1
 0 <= w_k1_INTC_s111 <= 1
 0 <= w_k1_INTC_s112 <= 1
 0 <= w_k1_INTC_s113 <= 1
 0 <= w_k1_INTC_s114 <= 1
 0 <= w_k1_INTC_s115 <= 1
 0 <= w_k1_INTC_s116 <= 1
 0 <= w_k1_INTC_s117 <= 1
 0 <= w_k1_KO_s0 <= 1
 0 <= w_k1_KO_s1 <= 1
 0 <= w_k1_KO_s00 <= 1
 0 <= w_k1_KO_s01 <= 1
 0 <= w_k1_KO_s10 <= 1
 0 <= w_k1_KO_s11 <= 1
 0 <= w_k1_KO_s000 <= 1
 0 <= w_k1_KO_s001 <= 1
 0 <= w_k1_KO_s002 <= 1
 0 <= w_k1_KO_s003 <= 1
 0 <= w_k1_KO_s004 <= 1
 0 <= w_k1_KO_s005 <= 1
 0 <= w_k1_KO_s006 <= 1
 0 <= w_k1_KO_s007 <= 1
 0 <= w_k1_KO_s010 <= 1
 0 <= w_k1_KO_s011 <= 1
 0 <= w_k1_KO_s012 <= 1
 0 <= w_k1_KO_s013 <= 1
 0 <= w_k1_KO_s014 <= 1
 0 <= w_k1_KO_s015 <= 1
 0 <= w_k1_KO_s016 <= 1
 0 <= w_k1_KO_s017 <= 1
 0 <= w_k1_KO_s100 <= 1
 0 <= w_k1_KO_s101 <= 1
 0 <= w_k1_KO_s102 <= 1
 0 <= w_k1_KO_s103 <= 1
 0 <= w_k1_KO_s104 <= 1
 0 <= w_k1_KO_s105 <= 1
 0 <= w_k1_KO_s106 <= 1
 0 <= w_k1_KO_s107 <= 1
 0 <= w_k1_KO_s110 <= 1
 0 <= w_k1_KO_s111 <= 1
 0 <= w_k1_KO_s112 <= 1
 0 <= w_k1_KO_s113 <= 1
 0 <= w_k1_KO_s114 <= 1
 0 <= w_k1_KO_s115 <= 1
 0 <= w_k1_KO_s116 <= 1
 0 <= w_k1_KO_s117 <= 1
 0 <= w_k1_MAT_s0 <= 1
 0 <= w_k1_MAT_s1 <= 1
 0 <= w_k1_MAT_s00 <= 1
 0 <= w_k1_MAT_s01 <= 1
 0 <= w_k1_MAT_s10 <= 1
 0 <= w_k1_MAT_s11 <= 1
 0 <= w_k1_MAT_s000 <= 1
 0 <= w_k1_MAT_s001 <= 1
 0 <= w_k1_MAT_s002 <= 1
 0 <= w_k1_MAT_s003 <= 1
 0 <= w_k1_MAT_s004 <= 1
 0 <= w_k1_MAT_s005 <= 1
 0 <= w_k1_MAT_s006 <= 1
 0 <= w_k1_MAT_s007 <= 1
 0 <= w_k1_MAT_s010 <= 1
 0 <= w_k1_MAT_s011 <= 1
 0 <= w_k1_MAT_s012 <= 1
 0 <= w_k1_MAT_s013 <= 1
 0 <= w_k1_MAT_s014 <= 1
 0 <= w_k1_MAT_s015 <= 1
 0 <= w_k1_MAT_s016 <= 1
 0 <= w_k1_MAT_s017 <= 1
 0 <= w_k1_MAT_s100 <= 1
 0 <= w_k1_MAT_s101 <= 1
 0 <= w_k1_MAT_s102 <= 1
 0 <= w_k1_MAT_s103 <= 1
 0 <= w_k1_MAT_s104 <= 1
 0 <= w_k1_MAT_s105 <= 1
 0 <= w_k1_MAT_s106 <= 1
 0 <= w_k1_MAT_s107 <= 1
 0 <= w_k1_MAT_s110 <= 1
 0 <= w_k1_MAT_s111 <= 1
 0 <= w_k1_MAT_s112 <= 1
 0 <= w_k1_MAT_s113 <= 1
 0 <= w_k1_MAT_s114 <= 1
 0 <= w_k1_MAT_s115 <= 1
 0 <= w_k1_MAT_s116 <= 1
 0 <= w_k1_MAT_s117 <= 1
 0 <= w_k1_MSFT_s0 <= 1
 0 <= w_k1_MSFT_s1 <= 1
 0 <= w_k1_MSFT_s00 <= 1
 0 <= w_k1_MSFT_s01 <= 1
 0 <= w_k1_MSFT_s10 <= 1
 0 <= w_k1_MSFT_s11 <= 1
 0 <= w_k1_MSFT_s000 <= 1
 0 <= w_k1_MSFT_s001 <= 1
 0 <= w_k1_MSFT_s002 <= 1
 0 <= w_k1_MSFT_s003 <= 1
 0 <= w_k1_MSFT_s004 <= 1
 0 <= w_k1_MSFT_s005 <= 1
 0 <= w_k1_MSFT_s006 <= 1
 0 <= w_k1_MSFT_s007 <= 1
 0 <= w_k1_MSFT_s010 <= 1
 0 <= w_k1_MSFT_s011 <= 1
 0 <= w_k1_MSFT_s012 <= 1
 0 <= w_k1_MSFT_s013 <= 1
 0 <= w_k1_MSFT_s014 <= 1
 0 <= w_k1_MSFT_s015 <= 1
 0 <= w_k1_MSFT_s016 <= 1
 0 <= w_k1_MSFT_s017 <= 1
 0 <= w_k1_MSFT_s100 <= 1
 0 <= w_k1_MSFT_s101 <= 1
 0 <= w_k1_MSFT_s102 <= 1
 0 <= w_k1_MSFT_s103 <= 1
 0 <= w_k1_MSFT_s104 <= 1
 0 <= w_k1_MSFT_s105 <= 1
 0 <= w_k1_MSFT_s106 <= 1
 0 <= w_k1_MSFT_s107 <= 1
 0 <= w_k1_MSFT_s110 <= 1
 0 <= w_k1_MSFT_s111 <= 1
 0 <= w_k1_MSFT_s112 <= 1
 0 <= w_k1_MSFT_s113 <= 1
 0 <= w_k1_MSFT_s114 <= 1
 0 <= w_k1_MSFT_s115 <= 1
 0 <= w_k1_MSFT_s116 <= 1
 0 <= w_k1_MSFT_s117 <= 1
 0 <= w_k1_PG_s0 <= 1
 0 <= w_k1_PG_s1 <= 1
 0 <= w_k1_PG_s00 <= 1
 0 <= w_k1_PG_s01 <= 1
 0 <= w_k1_PG_s10 <= 1
 0 <= w_k1_PG_s11 <= 1
 0 <= w_k1_PG_s000 <= 1
 0 <= w_k1_PG_s001 <= 1
 0 <= w_k1_PG_s002 <= 1
 0 <= w_k1_PG_s003 <= 1
 0 <= w_k1_PG_s004 <= 1
 0 <= w_k1_PG_s005 <= 1
 0 <= w_k1_PG_s006 <= 1
 0 <= w_k1_PG_s007 <= 1
 0 <= w_k1_PG_s010 <= 1
 0 <= w_k1_PG_s011 <= 1
 0 <= w_k1_PG_s012 <= 1
 0 <= w_k1_PG_s013 <= 1
 0 <= w_k1_PG_s014 <= 1
 0 <= w_k1_PG_s015 <= 1
 0 <= w_k1_PG_s016 <= 1
 0 <= w_k1_PG_s017 <= 1
 0 <= w_k1_PG_s100 <= 1
 0 <= w_k1_PG_s101 <= 1
 0 <= w_k1_PG_s102 <= 1
 0 <= w_k1_PG_s103 <= 1
 0 <= w_k1_PG_s104 <= 1
 0 <= w_k1_PG_s105 <= 1
 0 <= w_k1_PG_s106 <= 1
 0 <= w_k1_PG_s107 <= 1
 0 <= w_k1_PG_s110 <= 1
 0 <= w_k1_PG_s111 <= 1
 0 <= w_k1_PG_s112 <= 1
 0 <= w_k1_PG_s113 <= 1
 0 <= w_k1_PG_s114 <= 1
 0 <= w_k1_PG_s115 <= 1
 0 <= w_k1_PG_s116 <= 1
 0 <= w_k1_PG_s117 <= 1
 0 <= b_0_ADM <= 0.2
 0 <= b_0_AXP <= 0.2
 0 <= b_0_DIS <= 0.2
 0 <= b_0_FDX <= 0.2
 0 <= b_0_IBM <= 0.2
 0 <= b_0_INTC <= 0.2
 0 <= b_0_KO <= 0.2
 0 <= b_0_MAT <= 0.2
 0 <= b_0_MSFT <= 0.2
 0 <= b_0_PG <= 0.2
 0 <= b_k1_ADM_s0 <= 0.2
 0 <= b_k1_ADM_s1 <= 0.2
 0 <= b_k1_ADM_s00 <= 0.2
 0 <= b_k1_ADM_s01 <= 0.2
 0 <= b_k1_ADM_s10 <= 0.2
 0 <= b_k1_ADM_s11 <= 0.2
 0 <= b_k1_ADM_s000 <= 0.2
 0 <= b_k1_ADM_s001 <= 0.2
 0 <= b_k1_ADM_s002 <= 0.2
 0 <= b_k1_ADM_s003 <= 0.2
 0 <= b_k1_ADM_s004 <= 0.2
 0 <= b_k1_ADM_s005 <= 0.2
 0 <= b_k1_ADM_s006 <= 0.2
 0 <= b_k1_ADM_s007 <= 0.2
 0 <= b_k1_ADM_s010 <= 0.2
 0 <= b_k1_ADM_s011 <= 0.2
 0 <= b_k1_ADM_s012 <= 0.2
 0 <= b_k1_ADM_s013 <= 0.2
 0 <= b_k1_ADM_s014 <= 0.2
 0 <= b_k1_ADM_s015 <= 0.2
 0 <= b_k1_ADM_s016 <= 0.2
 0 <= b_k1_ADM_s017 <= 0.2
 0 <= b_k1_ADM_s100 <= 0.2
 0 <= b_k1_ADM_s101 <= 0.2
 0 <= b_k1_ADM_s102 <= 0.2
 0 <= b_k1_ADM_s103 <= 0.2
 0 <= b_k1_ADM_s104 <= 0.2
 0 <= b_k1_ADM_s105 <= 0.2
 0 <= b_k1_ADM_s106 <= 0.2
 0 <= b_k1_ADM_s107 <= 0.2
 0 <= b_k1_ADM_s110 <= 0.2
 0 <= b_k1_ADM_s111 <= 0.2
 0 <= b_k1_ADM_s112 <= 0.2
 0 <= b_k1_ADM_s113 <= 0.2
 0 <= b_k1_ADM_s114 <= 0.2
 0 <= b_k1_ADM_s115 <= 0.2
 0 <= b_k1_ADM_s116 <= 0.2
 0 <= b_k1_ADM_s117 <= 0.2
 0 <= b_k1_AXP_s0 <= 0.2
 0 <= b_k1_AXP_s1 <= 0.2
 0 <= b_k1_AXP_s00 <= 0.2
 0 <= b_k1_AXP_s01 <= 0.2
 0 <= b_k1_AXP_s10 <= 0.2
 0 <= b_k1_AXP_s11 <= 0.2
 0 <= b_k1_AXP_s000 <= 0.2
 0 <= b_k1_AXP_s001 <= 0.2
 0 <= b_k1_AXP_s002 <= 0.2
 0 <= b_k1_AXP_s003 <= 0.2
 0 <= b_k1_AXP_s004 <= 0.2
 0 <= b_k1_AXP_s005 <= 0.2
 0 <= b_k1_AXP_s006 <= 0.2
 0 <= b_k1_AXP_s007 <= 0.2
 0 <= b_k1_AXP_s010 <= 0.2
 0 <= b_k1_AXP_s011 <= 0.2
 0 <= b_k1_AXP_s012 <= 0.2
 0 <= b_k1_AXP_s013 <= 0.2
 0 <= b_k1_AXP_s014 <= 0.2
 0 <= b_k1_AXP_s015 <= 0.2
 0 <= b_k1_AXP_s016 <= 0.2
 0 <= b_k1_AXP_s017 <= 0.2
 0 <= b_k1_AXP_s100 <= 0.2
 0 <= b_k1_AXP_s101 <= 0.2
 0 <= b_k1_AXP_s102 <= 0.2
 0 <= b_k1_AXP_s103 <= 0.2
 0 <= b_k1_AXP_s104 <= 0.2
 0 <= b_k1_AXP_s105 <= 0.2
 0 <= b_k1_AXP_s106 <= 0.2
 0 <= b_k1_AXP_s107 <= 0.2
 0 <= b_k1_AXP_s110 <= 0.2
 0 <= b_k1_AXP_s111 <= 0.2
 0 <= b_k1_AXP_s112 <= 0.2
 0 <= b_k1_AXP_s113 <= 0.2
 0 <= b_k1_AXP_s114 <= 0.2
 0 <= b_k1_AXP_s115 <= 0.2
 0 <= b_k1_AXP_s116 <= 0.2
 0 <= b_k1_AXP_s117 <= 0.2
 0 <= b_k1_DIS_s0 <= 0.2
 0 <= b_k1_DIS_s1 <= 0.2
 0 <= b_k1_DIS_s00 <= 0.2
 0 <= b_k1_DIS_s01 <= 0.2
 0 <= b_k1_DIS_s10 <= 0.2
 0 <= b_k1_DIS_s11 <= 0.2
 0 <= b_k1_DIS_s000 <= 0.2
 0 <= b_k1_DIS_s001 <= 0.2
 0 <= b_k1_DIS_s002 <= 0.2
 0 <= b_k1_DIS_s003 <= 0.2
 0 <= b_k1_DIS_s004 <= 0.2
 0 <= b_k1_DIS_s005 <= 0.2
 0 <= b_k1_DIS_s006 <= 0.2
 0 <= b_k1_DIS_s007 <= 0.2
 0 <= b_k1_DIS_s010 <= 0.2
 0 <= b_k1_DIS_s011 <= 0.2
 0 <= b_k1_DIS_s012 <= 0.2
 0 <= b_k1_DIS_s013 <= 0.2
 0 <= b_k1_DIS_s014 <= 0.2
 0 <= b_k1_DIS_s015 <= 0.2
 0 <= b_k1_DIS_s016 <= 0.2
 0 <= b_k1_DIS_s017 <= 0.2
 0 <= b_k1_DIS_s100 <= 0.2
 0 <= b_k1_DIS_s101 <= 0.2
 0 <= b_k1_DIS_s102 <= 0.2
 0 <= b_k1_DIS_s103 <= 0.2
 0 <= b_k1_DIS_s104 <= 0.2
 0 <= b_k1_DIS_s105 <= 0.2
 0 <= b_k1_DIS_s106 <= 0.2
 0 <= b_k1_DIS_s107 <= 0.2
 0 <= b_k1_DIS_s110 <= 0.2
 0 <= b_k1_DIS_s111 <= 0.2
 0 <= b_k1_DIS_s112 <= 0.2
 0 <= b_k1_DIS_s113 <= 0.2
 0 <= b_k1_DIS_s114 <= 0.2
 0 <= b_k1_DIS_s115 <= 0.2
 0 <= b_k1_DIS_s116 <= 0.2
 0 <= b_k1_DIS_s117 <= 0.2
 0 <= b_k1_FDX_s0 <= 0.2
 0 <= b_k1_FDX_s1 <= 0.2
 0 <= b_k1_FDX_s00 <= 0.2
 0 <= b_k1_FDX_s01 <= 0.2
 0 <= b_k1_FDX_s10 <= 0.2
 0 <= b_k1_FDX_s11 <= 0.2
 0 <= b_k1_FDX_s000 <= 0.2
 0 <= b_k1_FDX_s001 <= 0.2
 0 <= b_k1_FDX_s002 <= 0.2
 0 <= b_k1_FDX_s003 <= 0.2
 0 <= b_k1_FDX_s004 <= 0.2
 0 <= b_k1_FDX_s005 <= 0.2
 0 <= b_k1_FDX_s006 <= 0.2
 0 <= b_k1_FDX_s007 <= 0.2
 0 <= b_k1_FDX_s010 <= 0.2
 0 <= b_k1_FDX_s011 <= 0.2
 0 <= b_k1_FDX_s012 <= 0.2
 0 <= b_k1_FDX_s013 <= 0.2
 0 <= b_k1_FDX_s014 <= 0.2
 0 <= b_k1_FDX_s015 <= 0.2
 0 <= b_k1_FDX_s016 <= 0.2
 0 <= b_k1_FDX_s017 <= 0.2
 0 <= b_k1_FDX_s100 <= 0.2
 0 <= b_k1_FDX_s101 <= 0.2
 0 <= b_k1_FDX_s102 <= 0.2
 0 <= b_k1_FDX_s103 <= 0.2
 0 <= b_k1_FDX_s104 <= 0.2
 0 <= b_k1_FDX_s105 <= 0.2
 0 <= b_k1_FDX_s106 <= 0.2
 0 <= b_k1_FDX_s107 <= 0.2
 0 <= b_k1_FDX_s110 <= 0.2
 0 <= b_k1_FDX_s111 <= 0.2
 0 <= b_k1_FDX_s112 <= 0.2
 0 <= b_k1_FDX_s113 <= 0.2
 0 <= b_k1_FDX_s114 <= 0.2
 0 <= b_k1_FDX_s115 <= 0.2
 0 <= b_k1_FDX_s116 <= 0.2
 0 <= b_k1_FDX_s117 <= 0.2
 0 <= b_k1_IBM_s0 <= 0.2
 0 <= b_k1_IBM_s1 <= 0.2
 0 <= b_k1_IBM_s00 <= 0.2
 0 <= b_k1_IBM_s01 <= 0.2
 0 <= b_k1_IBM_s10 <= 0.2
 0 <= b_k1_IBM_s11 <= 0.2
 0 <= b_k1_IBM_s000 <= 0.2
 0 <= b_k1_IBM_s001 <= 0.2
 0 <= b_k1_IBM_s002 <= 0.2
 0 <= b_k1_IBM_s003 <= 0.2
 0 <= b_k1_IBM_s004 <= 0.2
 0 <= b_k1_IBM_s005 <= 0.2
 0 <= b_k1_IBM_s006 <= 0.2
 0 <= b_k1_IBM_s007 <= 0.2
 0 <= b_k1_IBM_s010 <= 0.2
 0 <= b_k1_IBM_s011 <= 0.2
 0 <= b_k1_IBM_s012 <= 0.2
 0 <= b_k1_IBM_s013 <= 0.2
 0 <= b_k1_IBM_s014 <= 0.2
 0 <= b_k1_IBM_s015 <= 0.2
 0 <= b_k1_IBM_s016 <= 0.2
 0 <= b_k1_IBM_s017 <= 0.2
 0 <= b_k1_IBM_s100 <= 0.2
 0 <= b_k1_IBM_s101 <= 0.2
 0 <= b_k1_IBM_s102 <= 0.2
 0 <= b_k1_IBM_s103 <= 0.2
 0 <= b_k1_IBM_s104 <= 0.2
 0 <= b_k1_IBM_s105 <= 0.2
 0 <= b_k1_IBM_s106 <= 0.2
 0 <= b_k1_IBM_s107 <= 0.2
 0 <= b_k1_IBM_s110 <= 0.2
 0 <= b_k1_IBM_s111 <= 0.2
 0 <= b_k1_IBM_s112 <= 0.2
 0 <= b_k1_IBM_s113 <= 0.2
 0 <= b_k1_IBM_s114 <= 0.2
 0 <= b_k1_IBM_s115 <= 0.2
 0 <= b_k1_IBM_s116 <= 0.2
 0 <= b_k1_IBM_s117 <= 0.2
 0 <= b_k1_INTC_s0 <= 0.2
 0 <= b_k1_INTC_s1 <= 0.2
 0 <= b_k1_INTC_s00 <= 0.2
 0 <= b_k1_INTC_s01 <= 0.2
 0 <= b_k1_INTC_s10 <= 0.2
 0 <= b_k1_INTC_s11 <= 0.2
 0 <= b_k1_INTC_s000 <= 0.2
 0 <= b_k1_INTC_s001 <= 0.2
 0 <= b_k1_INTC_s002 <= 0.2
 0 <= b_k1_INTC_s003 <= 0.2
 0 <= b_k1_INTC_s004 <= 0.2
 0 <= b_k1_INTC_s005 <= 0.2
 0 <= b_k1_INTC_s006 <= 0.2
 0 <= b_k1_INTC_s007 <= 0.2
 0 <= b_k1_INTC_s010 <= 0.2
 0 <= b_k1_INTC_s011 <= 0.2
 0 <= b_k1_INTC_s012 <= 0.2
 0 <= b_k1_INTC_s013 <= 0.2
 0 <= b_k1_INTC_s014 <= 0.2
 0 <= b_k1_INTC_s015 <= 0.2
 0 <= b_k1_INTC_s016 <= 0.2
 0 <= b_k1_INTC_s017 <= 0.2
 0 <= b_k1_INTC_s100 <= 0.2
 0 <= b_k1_INTC_s101 <= 0.2
 0 <= b_k1_INTC_s102 <= 0.2
 0 <= b_k1_INTC_s103 <= 0.2
 0 <= b_k1_INTC_s104 <= 0.2
 0 <= b_k1_INTC_s105 <= 0.2
 0 <= b_k1_INTC_s106 <= 0.2
 0 <= b_k1_INTC_s107 <= 0.2
 0 <= b_k1_INTC_s110 <= 0.2
 0 <= b_k1_INTC_s111 <= 0.2
 0 <= b_k1_INTC_s112 <= 0.2
 0 <= b_k1_INTC_s113 <= 0.2
 0 <= b_k1_INTC_s114 <= 0.2
 0 <= b_k1_INTC_s115 <= 0.2
 0 <= b_k1_INTC_s116 <= 0.2
 0 <= b_k1_INTC_s117 <= 0.2
 0 <= b_k1_KO_s0 <= 0.2
 0 <= b_k1_KO_s1 <= 0.2
 0 <= b_k1_KO_s00 <= 0.2
 0 <= b_k1_KO_s01 <= 0.2
 0 <= b_k1_KO_s10 <= 0.2
 0 <= b_k1_KO_s11 <= 0.2
 0 <= b_k1_KO_s000 <= 0.2
 0 <= b_k1_KO_s001 <= 0.2
 0 <= b_k1_KO_s002 <= 0.2
 0 <= b_k1_KO_s003 <= 0.2
 0 <= b_k1_KO_s004 <= 0.2
 0 <= b_k1_KO_s005 <= 0.2
 0 <= b_k1_KO_s006 <= 0.2
 0 <= b_k1_KO_s007 <= 0.2
 0 <= b_k1_KO_s010 <= 0.2
 0 <= b_k1_KO_s011 <= 0.2
 0 <= b_k1_KO_s012 <= 0.2
 0 <= b_k1_KO_s013 <= 0.2
 0 <= b_k1_KO_s014 <= 0.2
 0 <= b_k1_KO_s015 <= 0.2
 0 <= b_k1_KO_s016 <= 0.2
 0 <= b_k1_KO_s017 <= 0.2
 0 <= b_k1_KO_s100 <= 0.2
 0 <= b_k1_KO_s101 <= 0.2
 0 <= b_k1_KO_s102 <= 0.2
 0 <= b_k1_KO_s103 <= 0.2
 0 <= b_k1_KO_s104 <= 0.2
 0 <= b_k1_KO_s105 <= 0.2
 0 <= b_k1_KO_s106 <= 0.2
 0 <= b_k1_KO_s107 <= 0.2
 0 <= b_k1_KO_s110 <= 0.2
 0 <= b_k1_KO_s111 <= 0.2
 0 <= b_k1_KO_s112 <= 0.2
 0 <= b_k1_KO_s113 <= 0.2
 0 <= b_k1_KO_s114 <= 0.2
 0 <= b_k1_KO_s115 <= 0.2
 0 <= b_k1_KO_s116 <= 0.2
 0 <= b_k1_KO_s117 <= 0.2
 0 <= b_k1_MAT_s0 <= 0.2
 0 <= b_k1_MAT_s1 <= 0.2
 0 <= b_k1_MAT_s00 <= 0.2
 0 <= b_k1_MAT_s01 <= 0.2
 0 <= b_k1_MAT_s10 <= 0.2
 0 <= b_k1_MAT_s11 <= 0.2
 0 <= b_k1_MAT_s000 <= 0.2
 0 <= b_k1_MAT_s001 <= 0.2
 0 <= b_k1_MAT_s002 <= 0.2
 0 <= b_k1_MAT_s003 <= 0.2
 0 <= b_k1_MAT_s004 <= 0.2
 0 <= b_k1_MAT_s005 <= 0.2
 0 <= b_k1_MAT_s006 <= 0.2
 0 <= b_k1_MAT_s007 <= 0.2
 0 <= b_k1_MAT_s010 <= 0.2
 0 <= b_k1_MAT_s011 <= 0.2
 0 <= b_k1_MAT_s012 <= 0.2
 0 <= b_k1_MAT_s013 <= 0.2
 0 <= b_k1_MAT_s014 <= 0.2
 0 <= b_k1_MAT_s015 <= 0.2
 0 <= b_k1_MAT_s016 <= 0.2
 0 <= b_k1_MAT_s017 <= 0.2
 0 <= b_k1_MAT_s100 <= 0.2
 0 <= b_k1_MAT_s101 <= 0.2
 0 <= b_k1_MAT_s102 <= 0.2
 0 <= b_k1_MAT_s103 <= 0.2
 0 <= b_k1_MAT_s104 <= 0.2
 0 <= b_k1_MAT_s105 <= 0.2
 0 <= b_k1_MAT_s106 <= 0.2
 0 <= b_k1_MAT_s107 <= 0.2
 0 <= b_k1_MAT_s110 <= 0.2
 0 <= b_k1_MAT_s111 <= 0.2
 0 <= b_k1_MAT_s112 <= 0.2
 0 <= b_k1_MAT_s113 <= 0.2
 0 <= b_k1_MAT_s114 <= 0.2
 0 <= b_k1_MAT_s115 <= 0.2
 0 <= b_k1_MAT_s116 <= 0.2
 0 <= b_k1_MAT_s117 <= 0.2
 0 <= b_k1_MSFT_s0 <= 0.2
 0 <= b_k1_MSFT_s1 <= 0.2
 0 <= b_k1_MSFT_s00 <= 0.2
 0 <= b_k1_MSFT_s01 <= 0.2
 0 <= b_k1_MSFT_s10 <= 0.2
 0 <= b_k1_MSFT_s11 <= 0.2
 0 <= b_k1_MSFT_s000 <= 0.2
 0 <= b_k1_MSFT_s001 <= 0.2
 0 <= b_k1_MSFT_s002 <= 0.2
 0 <= b_k1_MSFT_s003 <= 0.2
 0 <= b_k1_MSFT_s004 <= 0.2
 0 <= b_k1_MSFT_s005 <= 0.2
 0 <= b_k1_MSFT_s006 <= 0.2
 0 <= b_k1_MSFT_s007 <= 0.2
 0 <= b_k1_MSFT_s010 <= 0.2
 0 <= b_k1_MSFT_s011 <= 0.2
 0 <= b_k1_MSFT_s012 <= 0.2
 0 <= b_k1_MSFT_s013 <= 0.2
 0 <= b_k1_MSFT_s014 <= 0.2
 0 <= b_k1_MSFT_s015 <= 0.2
 0 <= b_k1_MSFT_s016 <= 0.2
 0 <= b_k1_MSFT_s017 <= 0.2
 0 <= b_k1_MSFT_s100 <= 0.2
 0 <= b_k1_MSFT_s101 <= 0.2
 0 <= b_k1_MSFT_s102 <= 0.2
 0 <= b_k1_MSFT_s103 <= 0.2
 0 <= b_k1_MSFT_s104 <= 0.2
 0 <= b_k1_MSFT_s105 <= 0.2
 0 <= b_k1_MSFT_s106 <= 0.2
 0 <= b_k1_MSFT_s107 <= 0.2
 0 <= b_k1_MSFT_s110 <= 0.2
 0 <= b_k1_MSFT_s111 <= 0.2
 0 <= b_k1_MSFT_s112 <= 0.2
 0 <= b_k1_MSFT_s113 <= 0.2
 0 <= b_k1_MSFT_s114 <= 0.2
 0 <= b_k1_MSFT_s115 <= 0.2
 0 <= b_k1_MSFT_s116 <= 0.2
 0 <= b_k1_MSFT_s117 <= 0.2
 0 <= b_k1_PG_s0 <= 0.2
 0 <= b_k1_PG_s1 <= 0.2
 0 <= b_k1_PG_s00 <= 0.2
 0 <= b_k1_PG_s01 <= 0.2
 0 <= b_k1_PG_s10 <= 0.2
 0 <= b_k1_PG_s11 <= 0.2
 0 <= b_k1_PG_s000 <= 0.2
 0 <= b_k1_PG_s001 <= 0.2
 0 <= b_k1_PG_s002 <= 0.2
 0 <= b_k1_PG_s003 <= 0.2
 0 <= b_k1_PG_s004 <= 0.2
 0 <= b_k1_PG_s005 <= 0.2
 0 <= b_k1_PG_s006 <= 0.2
 0 <= b_k1_PG_s007 <= 0.2
 0 <= b_k1_PG_s010 <= 0.2
 0 <= b_k1_PG_s011 <= 0.2
 0 <= b_k1_PG_s012 <= 0.2
 0 <= b_k1_PG_s013 <= 0.2
 0 <= b_k1_PG_s014 <= 0.2
 0 <= b_k1_PG_s015 <= 0.2
 0 <= b_k1_PG_s016 <= 0.2
 0 <= b_k1_PG_s017 <= 0.2
 0 <= b_k1_PG_s100 <= 0.2
 0 <= b_k1_PG_s101 <= 0.2
 0 <= b_k1_PG_s102 <= 0.2
 0 <= b_k1_PG_s103 <= 0.2
 0 <= b_k1_PG_s104 <= 0.2
 0 <= b_k1_PG_s105 <= 0.2
 0 <= b_k1_PG_s106 <= 0.2
 0 <= b_k1_PG_s107 <= 0.2
 0 <= b_k1_PG_s110 <= 0.2
 0 <= b_k1_PG_s111 <= 0.2
 0 <= b_k1_PG_s112 <= 0.2
 0 <= b_k1_PG_s113 <= 0.2
 0 <= b_k1_PG_s114 <= 0.2
 0 <= b_k1_PG_s115 <= 0.2
 0 <= b_k1_PG_s116 <= 0.2
 0 <= b_k1_PG_s117 <= 0.2
 0 <= s_0_ADM <= 0.2
 0 <= s_0_AXP <= 0.2
 0 <= s_0_DIS <= 0.2
 0 <= s_0_FDX <= 0.2
 0 <= s_0_IBM <= 0.2
 0 <= s_0_INTC <= 0.2
 0 <= s_0_KO <= 0.2
 0 <= s_0_MAT <= 0.2
 0 <= s_0_MSFT <= 0.2
 0 <= s_0_PG <= 0.2
 0 <= s_k1_ADM_s0 <= 0.2
 0 <= s_k1_ADM_s1 <= 0.2
 0 <= s_k1_ADM_s00 <= 0.2
 0 <= s_k1_ADM_s01 <= 0.2
 0 <= s_k1_ADM_s10 <= 0.2
 0 <= s_k1_ADM_s11 <= 0.2
 0 <= s_k1_ADM_s000 <= 0.2
 0 <= s_k1_ADM_s001 <= 0.2
 0 <= s_k1_ADM_s002 <= 0.2
 0 <= s_k1_ADM_s003 <= 0.2
 0 <= s_k1_ADM_s004 <= 0.2
 0 <= s_k1_ADM_s005 <= 0.2
 0 <= s_k1_ADM_s006 <= 0.2
 0 <= s_k1_ADM_s007 <= 0.2
 0 <= s_k1_ADM_s010 <= 0.2
 0 <= s_k1_ADM_s011 <= 0.2
 0 <= s_k1_ADM_s012 <= 0.2
 0 <= s_k1_ADM_s013 <= 0.2
 0 <= s_k1_ADM_s014 <= 0.2
 0 <= s_k1_ADM_s015 <= 0.2
 0 <= s_k1_ADM_s016 <= 0.2
 0 <= s_k1_ADM_s017 <= 0.2
 0 <= s_k1_ADM_s100 <= 0.2
 0 <= s_k1_ADM_s101 <= 0.2
 0 <= s_k1_ADM_s102 <= 0.2
 0 <= s_k1_ADM_s103 <= 0.2
 0 <= s_k1_ADM_s104 <= 0.2
 0 <= s_k1_ADM_s105 <= 0.2
 0 <= s_k1_ADM_s106 <= 0.2
 0 <= s_k1_ADM_s107 <= 0.2
 0 <= s_k1_ADM_s110 <= 0.2
 0 <= s_k1_ADM_s111 <= 0.2
 0 <= s_k1_ADM_s112 <= 0.2
 0 <= s_k1_ADM_s113 <= 0.2
 0 <= s_k1_ADM_s114 <= 0.2
 0 <= s_k1_ADM_s115 <= 0.2
 0 <= s_k1_ADM_s116 <= 0.2
 0 <= s_k1_ADM_s117 <= 0.2
 0 <= s_k1_AXP_s0 <= 0.2
 0 <= s_k1_AXP_s1 <= 0.2
 0 <= s_k1_AXP_s00 <= 0.2
 0 <= s_k1_AXP_s01 <= 0.2
 0 <= s_k1_AXP_s10 <= 0.2
 0 <= s_k1_AXP_s11 <= 0.2
 0 <= s_k1_AXP_s000 <= 0.2
 0 <= s_k1_AXP_s001 <= 0.2
 0 <= s_k1_AXP_s002 <= 0.2
 0 <= s_k1_AXP_s003 <= 0.2
 0 <= s_k1_AXP_s004 <= 0.2
 0 <= s_k1_AXP_s005 <= 0.2
 0 <= s_k1_AXP_s006 <= 0.2
 0 <= s_k1_AXP_s007 <= 0.2
 0 <= s_k1_AXP_s010 <= 0.2
 0 <= s_k1_AXP_s011 <= 0.2
 0 <= s_k1_AXP_s012 <= 0.2
 0 <= s_k1_AXP_s013 <= 0.2
 0 <= s_k1_AXP_s014 <= 0.2
 0 <= s_k1_AXP_s015 <= 0.2
 0 <= s_k1_AXP_s016 <= 0.2
 0 <= s_k1_AXP_s017 <= 0.2
 0 <= s_k1_AXP_s100 <= 0.2
 0 <= s_k1_AXP_s101 <= 0.2
 0 <= s_k1_AXP_s102 <= 0.2
 0 <= s_k1_AXP_s103 <= 0.2
 0 <= s_k1_AXP_s104 <= 0.2
 0 <= s_k1_AXP_s105 <= 0.2
 0 <= s_k1_AXP_s106 <= 0.2
 0 <= s_k1_AXP_s107 <= 0.2
 0 <= s_k1_AXP_s110 <= 0.2
 0 <= s_k1_AXP_s111 <= 0.2
 0 <= s_k1_AXP_s112 <= 0.2
 0 <= s_k1_AXP_s113 <= 0.2
 0 <= s_k1_AXP_s114 <= 0.2
 0 <= s_k1_AXP_s115 <= 0.2
 0 <= s_k1_AXP_s116 <= 0.2
 0 <= s_k1_AXP_s117 <= 0.2
 0 <= s_k1_DIS_s0 <= 0.2
 0 <= s_k1_DIS_s1 <= 0.2
 0 <= s_k1_DIS_s00 <= 0.2
 0 <= s_k1_DIS_s01 <= 0.2
 0 <= s_k1_DIS_s10 <= 0.2
 0 <= s_k1_DIS_s11 <= 0.2
 0 <= s_k1_DIS_s000 <= 0.2
 0 <= s_k1_DIS_s001 <= 0.2
 0 <= s_k1_DIS_s002 <= 0.2
 0 <= s_k1_DIS_s003 <= 0.2
 0 <= s_k1_DIS_s004 <= 0.2
 0 <= s_k1_DIS_s005 <= 0.2
 0 <= s_k1_DIS_s006 <= 0.2
 0 <= s_k1_DIS_s007 <= 0.2
 0 <= s_k1_DIS_s010 <= 0.2
 0 <= s_k1_DIS_s011 <= 0.2
 0 <= s_k1_DIS_s012 <= 0.2
 0 <= s_k1_DIS_s013 <= 0.2
 0 <= s_k1_DIS_s014 <= 0.2
 0 <= s_k1_DIS_s015 <= 0.2
 0 <= s_k1_DIS_s016 <= 0.2
 0 <= s_k1_DIS_s017 <= 0.2
 0 <= s_k1_DIS_s100 <= 0.2
 0 <= s_k1_DIS_s101 <= 0.2
 0 <= s_k1_DIS_s102 <= 0.2
 0 <= s_k1_DIS_s103 <= 0.2
 0 <= s_k1_DIS_s104 <= 0.2
 0 <= s_k1_DIS_s105 <= 0.2
 0 <= s_k1_DIS_s106 <= 0.2
 0 <= s_k1_DIS_s107 <= 0.2
 0 <= s_k1_DIS_s110 <= 0.2
 0 <= s_k1_DIS_s111 <= 0.2
 0 <= s_k1_DIS_s112 <= 0.2
 0 <= s_k1_DIS_s113 <= 0.2
 0 <= s_k1_DIS_s114 <= 0.2
 0 <= s_k1_DIS_s115 <= 0.2
 0 <= s_k1_DIS_s116 <= 0.2
 0 <= s_k1_DIS_s117 <= 0.2
 0 <= s_k1_FDX_s0 <= 0.2
 0 <= s_k1_FDX_s1 <= 0.2
 0 <= s_k1_FDX_s00 <= 0.2
 0 <= s_k1_FDX_s01 <= 0.2
 0 <= s_k1_FDX_s10 <= 0.2
 0 <= s_k1_FDX_s11 <= 0.2
 0 <= s_k1_FDX_s000 <= 0.2
 0 <= s_k1_FDX_s001 <= 0.2
 0 <= s_k1_FDX_s002 <= 0.2
 0 <= s_k1_FDX_s003 <= 0.2
 0 <= s_k1_FDX_s004 <= 0.2
 0 <= s_k1_FDX_s005 <= 0.2
 0 <= s_k1_FDX_s006 <= 0.2
 0 <= s_k1_FDX_s007 <= 0.2
 0 <= s_k1_FDX_s010 <= 0.2
 0 <= s_k1_FDX_s011 <= 0.2
 0 <= s_k1_FDX_s012 <= 0.2
 0 <= s_k1_FDX_s013 <= 0.2
 0 <= s_k1_FDX_s014 <= 0.2
 0 <= s_k1_FDX_s015 <= 0.2
 0 <= s_k1_FDX_s016 <= 0.2
 0 <= s_k1_FDX_s017 <= 0.2
 0 <= s_k1_FDX_s100 <= 0.2
 0 <= s_k1_FDX_s101 <= 0.2
 0 <= s_k1_FDX_s102 <= 0.2
 0 <= s_k1_FDX_s103 <= 0.2
 0 <= s_k1_FDX_s104 <= 0.2
 0 <= s_k1_FDX_s105 <= 0.2
 0 <= s_k1_FDX_s106 <= 0.2
 0 <= s_k1_FDX_s107 <= 0.2
 0 <= s_k1_FDX_s110 <= 0.2
 0 <= s_k1_FDX_s111 <= 0.2
 0 <= s_k1_FDX_s112 <= 0.2
 0 <= s_k1_FDX_s113 <= 0.2
 0 <= s_k1_FDX_s114 <= 0.2
 0 <= s_k1_FDX_s115 <= 0.2
 0 <= s_k1_FDX_s116 <= 0.2
 0 <= s_k1_FDX_s117 <= 0.2
 0 <= s_k1_IBM_s0 <= 0.2
 0 <= s_k1_IBM_s1 <= 0.2
 0 <= s_k1_IBM_s00 <= 0.2
 0 <= s_k1_IBM_s01 <= 0.2
 0 <= s_k1_IBM_s10 <= 0.2
 0 <= s_k1_IBM_s11 <= 0.2
 0 <= s_k1_IBM_s000 <= 0.2
 0 <= s_k1_IBM_s001 <= 0.2
 0 <= s_k1_IBM_s002 <= 0.2
 0 <= s_k1_IBM_s003 <= 0.2
 0 <= s_k1_IBM_s004 <= 0.2
 0 <= s_k1_IBM_s005 <= 0.2
 0 <= s_k1_IBM_s006 <= 0.2
 0 <= s_k1_IBM_s007 <= 0.2
 0 <= s_k1_IBM_s010 <= 0.2
 0 <= s_k1_IBM_s011 <= 0.2
 0 <= s_k1_IBM_s012 <= 0.2
 0 <= s_k1_IBM_s013 <= 0.2
 0 <= s_k1_IBM_s014 <= 0.2
 0 <= s_k1_IBM_s015 <= 0.2
 0 <= s_k1_IBM_s016 <= 0.2
 0 <= s_k1_IBM_s017 <= 0.2
 0 <= s_k1_IBM_s100 <= 0.2
 0 <= s_k1_IBM_s101 <= 0.2
 0 <= s_k1_IBM_s102 <= 0.2
 0 <= s_k1_IBM_s103 <= 0.2
 0 <= s_k1_IBM_s104 <= 0.2
 0 <= s_k1_IBM_s105 <= 0.2
 0 <= s_k1_IBM_s106 <= 0.2
 0 <= s_k1_IBM_s107 <= 0.2
 0 <= s_k1_IBM_s110 <= 0.2
 0 <= s_k1_IBM_s111 <= 0.2
 0 <= s_k1_IBM_s112 <= 0.2
 0 <= s_k1_IBM_s113 <= 0.2
 0 <= s_k1_IBM_s114 <= 0.2
 0 <= s_k1_IBM_s115 <= 0.2
 0 <= s_k1_IBM_s116 <= 0.2
 0 <= s_k1_IBM_s117 <= 0.2
 0 <= s_k1_INTC_s0 <= 0.2
 0 <= s_k1_INTC_s1 <= 0.2
 0 <= s_k1_INTC_s00 <= 0.2
 0 <= s_k1_INTC_s01 <= 0.2
 0 <= s_k1_INTC_s10 <= 0.2
 0 <= s_k1_INTC_s11 <= 0.2
 0 <= s_k1_INTC_s000 <= 0.2
 0 <= s_k1_INTC_s001 <= 0.2
 0 <= s_k1_INTC_s002 <= 0.2
 0 <= s_k1_INTC_s003 <= 0.2
 0 <= s_k1_INTC_s004 <= 0.2
 0 <= s_k1_INTC_s005 <= 0.2
 0 <= s_k1_INTC_s006 <= 0.2
 0 <= s_k1_INTC_s007 <= 0.2
 0 <= s_k1_INTC_s010 <= 0.2
 0 <= s_k1_INTC_s011 <= 0.2
 0 <= s_k1_INTC_s012 <= 0.2
 0 <= s_k1_INTC_s013 <= 0.2
 0 <= s_k1_INTC_s014 <= 0.2
 0 <= s_k1_INTC_s015 <= 0.2
 0 <= s_k1_INTC_s016 <= 0.2
 0 <= s_k1_INTC_s017 <= 0.2
 0 <= s_k1_INTC_s100 <= 0.2
 0 <= s_k1_INTC_s101 <= 0.2
 0 <= s_k1_INTC_s102 <= 0.2
 0 <= s_k1_INTC_s103 <= 0.2
 0 <= s_k1_INTC_s104 <= 0.2
 0 <= s_k1_INTC_s105 <= 0.2
 0 <= s_k1_INTC_s106 <= 0.2
 0 <= s_k1_INTC_s107 <= 0.2
 0 <= s_k1_INTC_s110 <= 0.2
 0 <= s_k1_INTC_s111 <= 0.2
 0 <= s_k1_INTC_s112 <= 0.2
 0 <= s_k1_INTC_s113 <= 0.2
 0 <= s_k1_INTC_s114 <= 0.2
 0 <= s_k1_INTC_s115 <= 0.2
 0 <= s_k1_INTC_s116 <= 0.2
 0 <= s_k1_INTC_s117 <= 0.2
 0 <= s_k1_KO_s0 <= 0.2
 0 <= s_k1_KO_s1 <= 0.2
 0 <= s_k1_KO_s00 <= 0.2
 0 <= s_k1_KO_s01 <= 0.2
 0 <= s_k1_KO_s10 <= 0.2
 0 <= s_k1_KO_s11 <= 0.2
 0 <= s_k1_KO_s000 <= 0.2
 0 <= s_k1_KO_s001 <= 0.2
 0 <= s_k1_KO_s002 <= 0.2
 0 <= s_k1_KO_s003 <= 0.2
 0 <= s_k1_KO_s004 <= 0.2
 0 <= s_k1_KO_s005 <= 0.2
 0 <= s_k1_KO_s006 <= 0.2
 0 <= s_k1_KO_s007 <= 0.2
 0 <= s_k1_KO_s010 <= 0.2
 0 <= s_k1_KO_s011 <= 0.2
 0 <= s_k1_KO_s012 <= 0.2
 0 <= s_k1_KO_s013 <= 0.2
 0 <= s_k1_KO_s014 <= 0.2
 0 <= s_k1_KO_s015 <= 0.2
 0 <= s_k1_KO_s016 <= 0.2
 0 <= s_k1_KO_s017 <= 0.2
 0 <= s_k1_KO_s100 <= 0.2
 0 <= s_k1_KO_s101 <= 0.2
 0 <= s_k1_KO_s102 <= 0.2
 0 <= s_k1_KO_s103 <= 0.2
 0 <= s_k1_KO_s104 <= 0.2
 0 <= s_k1_KO_s105 <= 0.2
 0 <= s_k1_KO_s106 <= 0.2
 0 <= s_k1_KO_s107 <= 0.2
 0 <= s_k1_KO_s110 <= 0.2
 0 <= s_k1_KO_s111 <= 0.2
 0 <= s_k1_KO_s112 <= 0.2
 0 <= s_k1_KO_s113 <= 0.2
 0 <= s_k1_KO_s114 <= 0.2
 0 <= s_k1_KO_s115 <= 0.2
 0 <= s_k1_KO_s116 <= 0.2
 0 <= s_k1_KO_s117 <= 0.2
 0 <= s_k1_MAT_s0 <= 0.2
 0 <= s_k1_MAT_s1 <= 0.2
 0 <= s_k1_MAT_s00 <= 0.2
 0 <= s_k1_MAT_s01 <= 0.2
 0 <= s_k1_MAT_s10 <= 0.2
 0 <= s_k1_MAT_s11 <= 0.2
 0 <= s_k1_MAT_s000 <= 0.2
 0 <= s_k1_MAT_s001 <= 0.2
 0 <= s_k1_MAT_s002 <= 0.2
 0 <= s_k1_MAT_s003 <= 0.2
 0 <= s_k1_MAT_s004 <= 0.2
 0 <= s_k1_MAT_s005 <= 0.2
 0 <= s_k1_MAT_s006 <= 0.2
 0 <= s_k1_MAT_s007 <= 0.2
 0 <= s_k1_MAT_s010 <= 0.2
 0 <= s_k1_MAT_s011 <= 0.2
 0 <= s_k1_MAT_s012 <= 0.2
 0 <= s_k1_MAT_s013 <= 0.2
 0 <= s_k1_MAT_s014 <= 0.2
 0 <= s_k1_MAT_s015 <= 0.2
 0 <= s_k1_MAT_s016 <= 0.2
 0 <= s_k1_MAT_s017 <= 0.2
 0 <= s_k1_MAT_s100 <= 0.2
 0 <= s_k1_MAT_s101 <= 0.2
 0 <= s_k1_MAT_s102 <= 0.2
 0 <= s_k1_MAT_s103 <= 0.2
 0 <= s_k1_MAT_s104 <= 0.2
 0 <= s_k1_MAT_s105 <= 0.2
 0 <= s_k1_MAT_s106 <= 0.2
 0 <= s_k1_MAT_s107 <= 0.2
 0 <= s_k1_MAT_s110 <= 0.2
 0 <= s_k1_MAT_s111 <= 0.2
 0 <= s_k1_MAT_s112 <= 0.2
 0 <= s_k1_MAT_s113 <= 0.2
 0 <= s_k1_MAT_s114 <= 0.2
 0 <= s_k1_MAT_s115 <= 0.2
 0 <= s_k1_MAT_s116 <= 0.2
 0 <= s_k1_MAT_s117 <= 0.2
 0 <= s_k1_MSFT_s0 <= 0.2
 0 <= s_k1_MSFT_s1 <= 0.2
 0 <= s_k1_MSFT_s00 <= 0.2
 0 <= s_k1_MSFT_s01 <= 0.2
 0 <= s_k1_MSFT_s10 <= 0.2
 0 <= s_k1_MSFT_s11 <= 0.2
 0 <= s_k1_MSFT_s000 <= 0.2
 0 <= s_k1_MSFT_s001 <= 0.2
 0 <= s_k1_MSFT_s002 <= 0.2
 0 <= s_k1_MSFT_s003 <= 0.2
 0 <= s_k1_MSFT_s004 <= 0.2
 0 <= s_k1_MSFT_s005 <= 0.2
 0 <= s_k1_MSFT_s006 <= 0.2
 0 <= s_k1_MSFT_s007 <= 0.2
 0 <= s_k1_MSFT_s010 <= 0.2
 0 <= s_k1_MSFT_s011 <= 0.2
 0 <= s_k1_MSFT_s012 <= 0.2
 0 <= s_k1_MSFT_s013 <= 0.2
 0 <= s_k1_MSFT_s014 <= 0.2
 0 <= s_k1_MSFT_s015 <= 0.2
 0 <= s_k1_MSFT_s016 <= 0.2
 0 <= s_k1_MSFT_s017 <= 0.2
 0 <= s_k1_MSFT_s100 <= 0.2
 0 <= s_k1_MSFT_s101 <= 0.2
 0 <= s_k1_MSFT_s102 <= 0.2
 0 <= s_k1_MSFT_s103 <= 0.2
 0 <= s_k1_MSFT_s104 <= 0.2
 0 <= s_k1_MSFT_s105 <= 0.2
 0 <= s_k1_MSFT_s106 <= 0.2
 0 <= s_k1_MSFT_s107 <= 0.2
 0 <= s_k1_MSFT_s110 <= 0.2
 0 <= s_k1_MSFT_s111 <= 0.2
 0 <= s_k1_MSFT_s112 <= 0.2
 0 <= s_k1_MSFT_s113 <= 0.2
 0 <= s_k1_MSFT_s114 <= 0.2
 0 <= s_k1_MSFT_s115 <= 0.2
 0 <= s_k1_MSFT_s116 <= 0.2
 0 <= s_k1_MSFT_s117 <= 0.2
 0 <= s_k1_PG_s0 <= 0.2
 0 <= s_k1_PG_s1 <= 0.2
 0 <= s_k1_PG_s00 <= 0.2
 0 <= s_k1_PG_s01 <= 0.2
 0 <= s_k1_PG_s10 <= 0.2
 0 <= s_k1_PG_s11 <= 0.2
 0 <= s_k1_PG_s000 <= 0.2
 0 <= s_k1_PG_s001 <= 0.2
 0 <= s_k1_PG_s002 <= 0.2
 0 <= s_k1_PG_s003 <= 0.2
 0 <= s_k1_PG_s004 <= 0.2
 0 <= s_k1_PG_s005 <= 0.2
 0 <= s_k1_PG_s006 <= 0.2
 0 <= s_k1_PG_s007 <= 0.2
 0 <= s_k1_PG_s010 <= 0.2
 0 <= s_k1_PG_s011 <= 0.2
 0 <= s_k1_PG_s012 <= 0.2
 0 <= s_k1_PG_s013 <= 0.2
 0 <= s_k1_PG_s014 <= 0.2
 0 <= s_k1_PG_s015 <= 0.2
 0 <= s_k1_PG_s016 <= 0.2
 0 <= s_k1_PG_s017 <= 0.2
 0 <= s_k1_PG_s100 <= 0.2
 0 <= s_k1_PG_s101 <= 0.2
 0 <= s_k1_PG_s102 <= 0.2
 0 <= s_k1_PG_s103 <= 0.2
 0 <= s_k1_PG_s104 <= 0.2
 0 <= s_k1_PG_s105 <= 0.2
 0 <= s_k1_PG_s106 <= 0.2
 0 <= s_k1_PG_s107 <= 0.2
 0 <= s_k1_PG_s110 <= 0.2
 0 <= s_k1_PG_s111 <= 0.2
 0 <= s_k1_PG_s112 <= 0.2
 0 <= s_k1_PG_s113 <= 0.2
 0 <= s_k1_PG_s114 <= 0.2
 0 <= s_k1_PG_s115 <= 0.2
 0 <= s_k1_PG_s116 <= 0.2
 0 <= s_k1_PG_s117 <= 0.2
End
