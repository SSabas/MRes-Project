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
 w_k1_ADM_s0_return_constraint:  1.17222 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.937885 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.08256 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.880721 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.04027 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.875364 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.34562 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.00499 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.708772 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.08021 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.965345 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.25914 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.844612 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.15045 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.33165 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.940817 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.05681 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.973291 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.64168 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.724843 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.878544 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.898944 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.836458 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.859984 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.897607 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.14975 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.962533 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.26443 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.2487 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.36693 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.21173 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.91982 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.764049 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.23558 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.11022 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.777013 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.10656 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.09275 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.20846 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.856429 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.11719 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.866439 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.15217 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.854895 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.33696 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.04974 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.660943 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.08704 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.00286 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.53633 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.702916 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.48015 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.39699 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.844937 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.18824 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.745709 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.599343 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.864301 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.74285 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.07885 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.789538 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.914589 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.765252 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.18291 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.95915 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.4913 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.0662 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.49038 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.3009 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.857169 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.779956 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.0517 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.941817 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.767951 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.27062 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.935733 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.22679 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.821534 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.22842 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.838879 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.27037 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.755071 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.46635 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.914466 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.550813 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.23743 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.909665 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.13234 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.901572 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.76848 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.57472 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.973235 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.31607 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.864016 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.585992 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.79506 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.679965 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.882066 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.76335 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.867951 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.641052 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.942029 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.779351 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.27759 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.15588 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.66824 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.42044 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.882221 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.736205 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.2181 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.53532 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.683537 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.40394 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.0789 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.15162 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.877684 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.12115 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.837967 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.11659 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.880489 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.38164 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.06686 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.755593 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.09426 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.929869 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.15717 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.891625 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.36238 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.24917 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.929165 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.22696 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.02264 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.779606 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.826318 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.869554 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.988402 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.779759 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.860636 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.819771 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.13231 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.883472 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.14535 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.18647 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.40679 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.31483 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.946799 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.800467 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.24009 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.20778 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.689669 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.15232 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.04173 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.15589 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.919726 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.935856 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.10667 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.22918 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.794553 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.960196 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.691154 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.670824 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.14097 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.1862 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.4872 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.72448 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.84262 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.65338 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.860997 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.973697 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.588311 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.46696 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.0828 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.689097 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.767719 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.02595 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.41799 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.586243 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.915379 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.871651 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.76846 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.695511 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.1495 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.940031 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.76723 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.930012 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.616839 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.909797 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.38458 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.5622 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.99189 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.05756 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.84138 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.0577 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.981825 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.17859 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.921298 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.48289 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.13557 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.85556 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.24692 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.04718 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.20448 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.98727 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.60674 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.0064 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.926813 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.989135 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.922442 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.774103 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.908471 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.779008 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.936312 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.936055 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.928824 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.78569 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.06643 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.02 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.08767 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.19097 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.17271 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.2424 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.0295 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.972887 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.23525 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.20684 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.864469 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.37076 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.10259 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.1256 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.864212 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.07525 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.04376 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.22152 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.876602 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.08002 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.788236 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.68024 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.18233 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.04648 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.27056 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.831684 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.604 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.36838 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.829145 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.0687 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.774216 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.551407 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.941882 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.72729 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.79953 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.933818 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.03832 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.607416 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.02926 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.830182 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.46207 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.960916 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.22628 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.05445 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.886454 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.851369 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.839493 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.0493 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.907499 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.28254 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.02444 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.08889 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.898503 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.16887 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.29127 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.36671 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.833388 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.822169 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.59486 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.573427 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.13792 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.969839 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.84745 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.732552 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.40591 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.85273 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.729348 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.10181 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.463863 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.349307 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.966909 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.524059 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.842781 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.683056 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.19408 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.535386 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.911369 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.814442 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.88014 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.642697 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.994754 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.779107 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.697172 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.737503 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.528571 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.678873 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.16932 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.32281 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.84975 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.10227 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.907443 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.11639 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.967125 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.17484 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.879374 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.30817 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.00036 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.791231 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.20331 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.929589 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.0342 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.01945 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.35833 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.17213 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.921852 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.978495 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.00522 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.717416 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.846514 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.791614 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.761642 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.831268 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.844767 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.679936 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.944548 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.784852 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.952618 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.13289 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.18495 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.23505 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.978549 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.811274 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.17137 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.15413 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.780467 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.18838 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.978899 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.10948 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.938984 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.06279 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.893426 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.09939 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.930014 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.22505 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.916634 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.790501 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.12205 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.938333 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.14639 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.930904 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.34531 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.21685 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.970506 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.13242 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.921556 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.751127 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.955703 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.936879 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.943041 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.850353 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.932852 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.740672 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.04866 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.918175 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.19943 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.958057 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.12383 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.99 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.885369 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.814444 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.932896 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.03142 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.855551 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.15975 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.01022 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.347999604209165 z_k1_s0000
                                 - 0.478001386035405 z_k1_s0001
                                 - 0.104000016660279 z_k1_s0002
                                 - 0.355001343754487 z_k1_s0003
                                 - 0.169999982079279 z_k1_s0004
                                 - 0.272999750233083 z_k1_s0005
                                 - 0.197000117350764 z_k1_s0006
                                 - 0.422999458185248 z_k1_s0007
                                 - 0.299999518252437 z_k1_s0010
                                 - 0.566997285556208 z_k1_s0011
                                 - 0.670000237310789 z_k1_s0012
                                 - 0.365999287858256 z_k1_s0013
                                 - 0.81499976255835 z_k1_s0014
                                 - 0.624997095655232 z_k1_s0015
                                 - 0.343999088190278 z_k1_s0016
                                 - 0.460000781883611 z_k1_s0017
                                 - 0.706001786083037 z_k1_s0020
                                 - 0.294000403076818 z_k1_s0021
                                 - 0.549000225092032 z_k1_s0022
                                 - 1.0709985780634 z_k1_s0023
                                 - 1.01800193857252 z_k1_s0024
                                 - 0.586998197406534 z_k1_s0025
                                 - 0.282000008134625 z_k1_s0026
                                 - 1.11300248936296 z_k1_s0027
                                 - 0.179000026963157 z_k1_s0030
                                 - 0.43100140417345 z_k1_s0031
                                 - 0.377001285585624 z_k1_s0032
                                 - 0.301999747879582 z_k1_s0033
                                 - 0.144000041634059 z_k1_s0034
                                 - 0.432000948702612 z_k1_s0035
                                 - 0.549999377171338 z_k1_s0036
                                 - 0.331000269232546 z_k1_s0037
                                 - 1.65700355165208 z_k1_s0040
                                 - 0.421001379530193 z_k1_s0041
                                 - 1.38900253208381 z_k1_s0042
                                 - 0.515001012884392 z_k1_s0043
                                 - 0.387000796315157 z_k1_s0044
                                 - 0.560001270991225 z_k1_s0045
                                 - 0.852004110420741 z_k1_s0046
                                 - 0.688998517088622 z_k1_s0047
                                 - 0.434001106697678 z_k1_s0050
                                 - 0.150000065820627 z_k1_s0051
                                 - 0.155000172134724 z_k1_s0052
                                 - 0.314000116960498 z_k1_s0053
                                 - 0.657000277882341 z_k1_s0054
                                 - 0.22900013172218 z_k1_s0055
                                 - 0.370001099438225 z_k1_s0056
                                 - 0.294001253505243 z_k1_s0057
                                 - 0.57699993807562 z_k1_s0060
                                 - 0.167000186856688 z_k1_s0061
                                 - 0.231000052166895 z_k1_s0062
                                 - 0.203000254643778 z_k1_s0063
                                 - 0.220000159054261 z_k1_s0064
                                 - 0.261000874256659 z_k1_s0065
                                 - 0.694001561984619 z_k1_s0066
                                 - 0.199000247111879 z_k1_s0067
                                 - 0.358999042327134 z_k1_s0070
                                 - 0.256999209636168 z_k1_s0071
                                 - 0.73999853183717 z_k1_s0072
                                 - 0.209999522005324 z_k1_s0073
                                 - 0.281999098682199 z_k1_s0074
                                 - 0.296999226689244 z_k1_s0075
                                 - 0.234999411051355 z_k1_s0076
                                 - 0.862998673457617 z_k1_s0077
                                 - 0.540996788859313 z_k1_s0100
                                 - 0.456999147040883 z_k1_s0101
                                 - 0.869998933921146 z_k1_s0102
                                 - 0.717996422722935 z_k1_s0103
                                 - 0.246999271708508 z_k1_s0104
                                 - 0.583998764787382 z_k1_s0105
                                 - 1.10699550937262 z_k1_s0106
                                 - 0.284999418609202 z_k1_s0107
                                 - 0.342000783158439 z_k1_s0110
                                 - 0.157999774838536 z_k1_s0111
                                 - 0.62899866233728 z_k1_s0112
                                 - 0.414000544771036 z_k1_s0113
                                 - 0.449999148578263 z_k1_s0114
                                 - 0.19899987320665 z_k1_s0115
                                 - 0.182999841048357 z_k1_s0116
                                 - 0.178999769158831 z_k1_s0117
                                 - 0.227000315410768 z_k1_s0120
                                 - 0.669000187000912 z_k1_s0121
                                 - 0.455999326940916 z_k1_s0122
                                 - 0.292000508565574 z_k1_s0123
                                 - 0.266000637284015 z_k1_s0124
                                 - 0.575001048483663 z_k1_s0125
                                 - 0.540001551855221 z_k1_s0126
                                 - 0.407999036420183 z_k1_s0127
                                 - 0.342999495110611 z_k1_s0130
                                 - 0.296999649410543 z_k1_s0131
                                 - 0.100999860776 z_k1_s0132
                                 - 0.0739998926477684 z_k1_s0133
                                 - 0.0729999488281394 z_k1_s0134
                                 - 0.125999806265132 z_k1_s0135
                                 - 0.0719998700086697 z_k1_s0136
                                 - 0.264000288364219 z_k1_s0137
                                 - 0.097999931779974 z_k1_s0140
                                 - 0.276998800361157 z_k1_s0141
                                 - 0.109999853183744 z_k1_s0142
                                 - 0.134999812366424 z_k1_s0143
                                 - 0.165999322501573 z_k1_s0144
                                 - 0.294999911715061 z_k1_s0145
                                 - 0.224999960443651 z_k1_s0146
                                 - 0.333000072576417 z_k1_s0147
                                 - 1.02599962602317 z_k1_s0150
                                 - 0.4719995597106 z_k1_s0151
                                 - 0.516998006993093 z_k1_s0152
                                 - 0.668999434255037 z_k1_s0153
                                 - 0.435000002389859 z_k1_s0154
                                 - 0.28499967371766 z_k1_s0155
                                 - 0.93599772746071 z_k1_s0156
                                 - 0.664000440778783 z_k1_s0157
                                 - 0.143000009009834 z_k1_s0160
                                 - 0.199999294699375 z_k1_s0161
                                 - 0.185000321096506 z_k1_s0162
                                 - 0.227000633183179 z_k1_s0163
                                 - 0.390999913236727 z_k1_s0164
                                 - 0.0839999161736468 z_k1_s0165
                                 - 0.282999869299245 z_k1_s0166
                                 - 0.257000350388161 z_k1_s0167
                                 - 0.352999315989604 z_k1_s0170
                                 - 0.0789999312593481 z_k1_s0171
                                 - 0.153999902194137 z_k1_s0172
                                 - 0.169999575292536 z_k1_s0173
                                 - 0.154999850225318 z_k1_s0174
                                 - 0.228999032129741 z_k1_s0175
                                 - 0.29599975521476 z_k1_s0176
                                 - 0.245998989658995 z_k1_s0177
                                 - 0.531997094299102 z_k1_s1000
                                 - 0.636997533697534 z_k1_s1001
                                 - 0.512998481070706 z_k1_s1002
                                 - 1.16299533602905 z_k1_s1003
                                 - 0.305999301052148 z_k1_s1004
                                 - 0.519998510363934 z_k1_s1005
                                 - 1.03899628340222 z_k1_s1006
                                 - 0.421999126655989 z_k1_s1007
                                 - 0.720996901310918 z_k1_s1010
                                 - 0.278998671436286 z_k1_s1011
                                 - 0.170999244770364 z_k1_s1012
                                 - 0.722995297400724 z_k1_s1013
                                 - 0.331999186724344 z_k1_s1014
                                 - 0.257998732068139 z_k1_s1015
                                 - 0.197999253986164 z_k1_s1016
                                 - 0.368998498756528 z_k1_s1017
                                 - 0.355999971645086 z_k1_s1020
                                 - 0.112000028769229 z_k1_s1021
                                 - 0.35300018052442 z_k1_s1022
                                 - 0.358000631058629 z_k1_s1023
                                 - 0.438000651608079 z_k1_s1024
                                 - 0.336000565507548 z_k1_s1025
                                 - 0.322000741611342 z_k1_s1026
                                 - 0.120999881331086 z_k1_s1027
                                 - 0.310001001636616 z_k1_s1030
                                 - 0.181000553871281 z_k1_s1031
                                 - 0.331999633834681 z_k1_s1032
                                 - 0.374000903143213 z_k1_s1033
                                 - 0.788001124288595 z_k1_s1034
                                 - 0.684002235680871 z_k1_s1035
                                 - 0.336000227436983 z_k1_s1036
                                 - 0.30400110393629 z_k1_s1037
                                 - 0.545002357071965 z_k1_s1040
                                 - 0.593000609367878 z_k1_s1041
                                 - 0.153000512873887 z_k1_s1042
                                 - 0.224000644840538 z_k1_s1043
                                 - 0.573001602662039 z_k1_s1044
                                 - 0.398001546209829 z_k1_s1045
                                 - 0.210000703944551 z_k1_s1046
                                 - 0.48500215594495 z_k1_s1047
                                 - 0.222000031965863 z_k1_s1050
                                 - 0.179999785972481 z_k1_s1051
                                 - 0.142999973644709 z_k1_s1052
                                 - 0.38899896060758 z_k1_s1053
                                 - 0.151999943643345 z_k1_s1054
                                 - 0.222000031965863 z_k1_s1055
                                 - 0.115999870648927 z_k1_s1056
                                 - 0.505999342589345 z_k1_s1057
                                 - 1.14599699990532 z_k1_s1060
                                 - 0.345999419362124 z_k1_s1061
                                 - 0.340999608018429 z_k1_s1062
                                 - 0.263999696530396 z_k1_s1063
                                 - 0.573000511949174 z_k1_s1064
                                 - 0.358999331255035 z_k1_s1065
                                 - 0.528000197859399 z_k1_s1066
                                 - 0.467000487469799 z_k1_s1067
                                 - 0.164999636855313 z_k1_s1070
                                 - 0.416998057533795 z_k1_s1071
                                 - 0.648997621525415 z_k1_s1072
                                 - 0.464999977736454 z_k1_s1073
                                 - 0.46399912798324 z_k1_s1074
                                 - 0.416000446776547 z_k1_s1075
                                 - 0.307999689216127 z_k1_s1076
                                 - 0.355000435765976 z_k1_s1077
                                 - 0.193999479862889 z_k1_s1100
                                 - 0.366998936019154 z_k1_s1101
                                 - 0.249999145081467 z_k1_s1102
                                 - 0.209999520667883 z_k1_s1103
                                 - 0.65699937711036 z_k1_s1104
                                 - 0.31100046479783 z_k1_s1105
                                 - 0.240998793779412 z_k1_s1106
                                 - 0.157999746250828 z_k1_s1107
                                 - 0.29100077705618 z_k1_s1110
                                 - 0.275000860359332 z_k1_s1111
                                 - 0.284001021876969 z_k1_s1112
                                 - 0.524003106340248 z_k1_s1113
                                 - 0.338001990982792 z_k1_s1114
                                 - 0.306000823984871 z_k1_s1115
                                 - 0.505002580135761 z_k1_s1116
                                 - 0.811001403714293 z_k1_s1117
                                 - 1.41400026566896 z_k1_s1120
                                 - 0.664000378040274 z_k1_s1121
                                 - 0.416000803187874 z_k1_s1122
                                 - 1.0780045242122 z_k1_s1123
                                 - 0.295000478095054 z_k1_s1124
                                 - 0.29000044153745 z_k1_s1125
                                 - 0.445001015221973 z_k1_s1126
                                 - 0.994002790842096 z_k1_s1127
                                 - 0.249999991669948 z_k1_s1130
                                 - 0.10999992909478 z_k1_s1131
                                 - 0.105999989744058 z_k1_s1132
                                 - 0.308999727468066 z_k1_s1133
                                 - 0.402999562959972 z_k1_s1134
                                 - 0.178999597319698 z_k1_s1135
                                 - 0.129999962048374 z_k1_s1136
                                 - 0.193999159759911 z_k1_s1137
                                 - 0.146999756435629 z_k1_s1140
                                 - 0.100000023167052 z_k1_s1141
                                 - 0.29199932732796 z_k1_s1142
                                 - 0.132999779632236 z_k1_s1143
                                 - 0.18600006953071 z_k1_s1144
                                 - 0.169000475412218 z_k1_s1145
                                 - 0.234000556570786 z_k1_s1146
                                 - 0.0609999744719106 z_k1_s1147
                                 - 0.20200080667701 z_k1_s1150
                                 - 0.181999787005095 z_k1_s1151
                                 - 0.274000409491905 z_k1_s1152
                                 - 0.0720000096152946 z_k1_s1153
                                 - 0.183000515989357 z_k1_s1154
                                 - 0.162999496317442 z_k1_s1155
                                 - 0.0910000291026807 z_k1_s1156
                                 - 0.189000821890932 z_k1_s1157
                                 - 0.699999512191746 z_k1_s1160
                                 - 0.219000302049267 z_k1_s1161
                                 - 0.400998959377231 z_k1_s1162
                                 - 0.487999598625007 z_k1_s1163
                                 - 0.547000114344892 z_k1_s1164
                                 - 0.523999246588646 z_k1_s1165
                                 - 0.35100068005187 z_k1_s1166
                                 - 0.346000709079111 z_k1_s1167
                                 - 0.452002782026756 z_k1_s1170
                                 - 0.180000836354615 z_k1_s1171
                                 - 0.194000895815506 z_k1_s1172
                                 - 0.121000442054528 z_k1_s1173
                                 - 0.534002196150794 z_k1_s1174
                                 - 0.473003122719178 z_k1_s1175
                                 - 0.167000745212205 z_k1_s1176
                                 - 0.39400160154189 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.959157 w_k1_ADM_s000
                                 + 1.51801 w_k1_AXP_s000
                                 + 1.13311 w_k1_DIS_s000 + 1.1967 w_k1_FDX_s000
                                 + 1.28066 w_k1_IBM_s000
                                 + 1.20151 w_k1_INTC_s000
                                 + 1.19608 w_k1_KO_s000 + 1.69559 w_k1_MAT_s000
                                 + 1.0766 w_k1_MSFT_s000
                                 + 0.978446 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.838148 w_k1_ADM_s000
                                 + 0.964843 w_k1_AXP_s000
                                 + 1.00842 w_k1_DIS_s000
                                 + 1.03697 w_k1_FDX_s000
                                 + 0.869235 w_k1_IBM_s000
                                 + 0.994805 w_k1_INTC_s000
                                 + 1.01016 w_k1_KO_s000 + 1.04963 w_k1_MAT_s000
                                 + 1.06289 w_k1_MSFT_s000
                                 + 0.821891 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.42118 w_k1_ADM_s000
                                 + 1.46936 w_k1_AXP_s000
                                 + 1.92071 w_k1_DIS_s000
                                 + 1.51956 w_k1_FDX_s000
                                 + 1.16701 w_k1_IBM_s000
                                 + 1.25653 w_k1_INTC_s000
                                 + 1.58532 w_k1_KO_s000 + 1.37715 w_k1_MAT_s000
                                 + 1.43579 w_k1_MSFT_s000 + 1.1428 w_k1_PG_s000
                                 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.837784 w_k1_ADM_s000
                                 + 1.03695 w_k1_AXP_s000
                                 + 0.843056 w_k1_DIS_s000
                                 + 0.868746 w_k1_FDX_s000
                                 + 1.31069 w_k1_IBM_s000
                                 + 0.856406 w_k1_INTC_s000
                                 + 1.01824 w_k1_KO_s000 + 1.30116 w_k1_MAT_s000
                                 + 0.825681 w_k1_MSFT_s000
                                 + 0.81356 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.0472 w_k1_ADM_s000
                                 + 1.13612 w_k1_AXP_s000
                                 + 1.29977 w_k1_DIS_s000
                                 + 1.30949 w_k1_FDX_s000
                                 + 0.765816 w_k1_IBM_s000
                                 + 1.00498 w_k1_INTC_s000
                                 + 1.03822 w_k1_KO_s000 + 1.02494 w_k1_MAT_s000
                                 + 1.09964 w_k1_MSFT_s000
                                 + 0.905715 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.00947 w_k1_ADM_s000
                                 + 1.01409 w_k1_AXP_s000
                                 + 1.33596 w_k1_DIS_s000
                                 + 1.06107 w_k1_FDX_s000
                                 + 1.27711 w_k1_IBM_s000
                                 + 0.970661 w_k1_INTC_s000
                                 + 1.33009 w_k1_KO_s000 + 0.95158 w_k1_MAT_s000
                                 + 1.16855 w_k1_MSFT_s000
                                 + 1.05994 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.689934 w_k1_ADM_s000
                                 + 0.73164 w_k1_AXP_s000
                                 + 0.559219 w_k1_DIS_s000
                                 + 0.748195 w_k1_FDX_s000
                                 + 0.842801 w_k1_IBM_s000
                                 + 0.661083 w_k1_INTC_s000
                                 + 0.794696 w_k1_KO_s000
                                 + 0.891001 w_k1_MAT_s000
                                 + 0.674321 w_k1_MSFT_s000
                                 + 0.70271 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.708307 w_k1_ADM_s000
                                 + 0.87079 w_k1_AXP_s000
                                 + 0.741225 w_k1_DIS_s000
                                 + 0.892769 w_k1_FDX_s000
                                 + 0.783925 w_k1_IBM_s000
                                 + 0.942954 w_k1_INTC_s000
                                 + 0.901111 w_k1_KO_s000
                                 + 0.869389 w_k1_MAT_s000
                                 + 0.927151 w_k1_MSFT_s000
                                 + 0.86468 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.878165 w_k1_ADM_s001
                                 + 0.768956 w_k1_AXP_s001
                                 + 0.629569 w_k1_DIS_s001
                                 + 0.778733 w_k1_FDX_s001
                                 + 1.08985 w_k1_IBM_s001
                                 + 0.771544 w_k1_INTC_s001
                                 + 0.927633 w_k1_KO_s001
                                 + 1.10662 w_k1_MAT_s001
                                 + 0.791224 w_k1_MSFT_s001
                                 + 0.837376 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.32391 w_k1_ADM_s001
                                 + 1.40296 w_k1_AXP_s001
                                 + 1.24231 w_k1_DIS_s001
                                 + 1.28125 w_k1_FDX_s001
                                 + 1.37354 w_k1_IBM_s001
                                 + 1.29107 w_k1_INTC_s001
                                 + 1.34482 w_k1_KO_s001 + 1.55819 w_k1_MAT_s001
                                 + 1.25307 w_k1_MSFT_s001
                                 + 1.33725 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.910575 w_k1_ADM_s001
                                 + 0.986341 w_k1_AXP_s001
                                 + 0.791356 w_k1_DIS_s001
                                 + 0.948971 w_k1_FDX_s001
                                 + 0.960433 w_k1_IBM_s001
                                 + 0.906558 w_k1_INTC_s001
                                 + 1.00489 w_k1_KO_s001 + 1.28781 w_k1_MAT_s001
                                 + 1.01142 w_k1_MSFT_s001 + 1.046 w_k1_PG_s001
                                 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.848877 w_k1_ADM_s001
                                 + 0.66031 w_k1_AXP_s001
                                 + 0.613788 w_k1_DIS_s001
                                 + 0.798004 w_k1_FDX_s001
                                 + 0.507537 w_k1_IBM_s001
                                 + 0.797106 w_k1_INTC_s001
                                 + 0.668777 w_k1_KO_s001
                                 + 0.75676 w_k1_MAT_s001
                                 + 0.820879 w_k1_MSFT_s001
                                 + 0.860219 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.07554 w_k1_ADM_s001
                                 + 0.911341 w_k1_AXP_s001
                                 + 0.972149 w_k1_DIS_s001
                                 + 1.08315 w_k1_FDX_s001
                                 + 0.769627 w_k1_IBM_s001
                                 + 1.01582 w_k1_INTC_s001
                                 + 1.02436 w_k1_KO_s001 + 1.06395 w_k1_MAT_s001
                                 + 1.09404 w_k1_MSFT_s001
                                 + 0.986279 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.76737 w_k1_ADM_s001
                                 + 0.927171 w_k1_AXP_s001
                                 + 1.02269 w_k1_DIS_s001
                                 + 0.941994 w_k1_FDX_s001
                                 + 1.1454 w_k1_IBM_s001
                                 + 1.02026 w_k1_INTC_s001
                                 + 1.26588 w_k1_KO_s001 + 1.28268 w_k1_MAT_s001
                                 + 1.12568 w_k1_MSFT_s001 + 1.0438 w_k1_PG_s001
                                 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.23682 w_k1_ADM_s001
                                 + 1.24278 w_k1_AXP_s001
                                 + 1.45338 w_k1_DIS_s001
                                 + 1.38052 w_k1_FDX_s001
                                 + 0.97696 w_k1_IBM_s001
                                 + 1.35577 w_k1_INTC_s001
                                 + 1.18878 w_k1_KO_s001
                                 + 0.887396 w_k1_MAT_s001
                                 + 1.25191 w_k1_MSFT_s001
                                 + 1.16387 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.957723 w_k1_ADM_s001
                                 + 1.00469 w_k1_AXP_s001
                                 + 0.849518 w_k1_DIS_s001
                                 + 0.881328 w_k1_FDX_s001
                                 + 1.37868 w_k1_IBM_s001
                                 + 0.854747 w_k1_INTC_s001
                                 + 1.14196 w_k1_KO_s001 + 1.75362 w_k1_MAT_s001
                                 + 0.853327 w_k1_MSFT_s001
                                 + 1.06147 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.33416 w_k1_ADM_s002
                                 + 1.27412 w_k1_AXP_s002
                                 + 1.57457 w_k1_DIS_s002
                                 + 1.38364 w_k1_FDX_s002
                                 + 1.05109 w_k1_IBM_s002
                                 + 1.24628 w_k1_INTC_s002
                                 + 1.22544 w_k1_KO_s002 + 1.12566 w_k1_MAT_s002
                                 + 1.2407 w_k1_MSFT_s002 + 1.22756 w_k1_PG_s002
                                 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.710433 w_k1_ADM_s002
                                 + 0.631597 w_k1_AXP_s002
                                 + 0.571075 w_k1_DIS_s002
                                 + 0.647284 w_k1_FDX_s002
                                 + 0.693654 w_k1_IBM_s002
                                 + 0.839833 w_k1_INTC_s002
                                 + 0.719936 w_k1_KO_s002
                                 + 0.713047 w_k1_MAT_s002
                                 + 0.77084 w_k1_MSFT_s002
                                 + 0.770536 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.19757 w_k1_ADM_s002
                                 + 1.19625 w_k1_AXP_s002
                                 + 1.56041 w_k1_DIS_s002
                                 + 1.18843 w_k1_FDX_s002 + 1.4586 w_k1_IBM_s002
                                 + 0.924722 w_k1_INTC_s002
                                 + 1.47921 w_k1_KO_s002 + 1.90126 w_k1_MAT_s002
                                 + 1.13879 w_k1_MSFT_s002
                                 + 1.19838 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.13594 w_k1_ADM_s002
                                 + 1.07201 w_k1_AXP_s002
                                 + 0.895984 w_k1_DIS_s002
                                 + 0.867087 w_k1_FDX_s002
                                 + 1.27118 w_k1_IBM_s002
                                 + 0.936901 w_k1_INTC_s002
                                 + 0.97599 w_k1_KO_s002 + 1.03361 w_k1_MAT_s002
                                 + 0.828623 w_k1_MSFT_s002
                                 + 0.995863 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.0152 w_k1_ADM_s002
                                 + 0.829393 w_k1_AXP_s002
                                 + 1.08013 w_k1_DIS_s002
                                 + 0.976553 w_k1_FDX_s002
                                 + 0.924778 w_k1_IBM_s002
                                 + 1.00627 w_k1_INTC_s002
                                 + 1.07554 w_k1_KO_s002
                                 + 0.928306 w_k1_MAT_s002
                                 + 1.07765 w_k1_MSFT_s002
                                 + 1.06523 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.18331 w_k1_ADM_s002
                                 + 1.07699 w_k1_AXP_s002
                                 + 1.22104 w_k1_DIS_s002
                                 + 0.919531 w_k1_FDX_s002
                                 + 1.82728 w_k1_IBM_s002
                                 + 0.852026 w_k1_INTC_s002
                                 + 1.3366 w_k1_KO_s002 + 1.66795 w_k1_MAT_s002
                                 + 0.930111 w_k1_MSFT_s002
                                 + 1.05777 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.06661 w_k1_ADM_s002
                                 + 0.718669 w_k1_AXP_s002
                                 + 0.608294 w_k1_DIS_s002
                                 + 0.755462 w_k1_FDX_s002
                                 + 0.844739 w_k1_IBM_s002
                                 + 0.76527 w_k1_INTC_s002
                                 + 0.768952 w_k1_KO_s002
                                 + 0.918381 w_k1_MAT_s002
                                 + 0.733039 w_k1_MSFT_s002
                                 + 0.857462 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.10966 w_k1_ADM_s002
                                 + 1.01155 w_k1_AXP_s002
                                 + 0.885391 w_k1_DIS_s002
                                 + 1.04246 w_k1_FDX_s002
                                 + 0.66532 w_k1_IBM_s002
                                 + 0.929007 w_k1_INTC_s002
                                 + 0.757046 w_k1_KO_s002
                                 + 0.687284 w_k1_MAT_s002
                                 + 0.911194 w_k1_MSFT_s002
                                 + 0.906996 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.25822 w_k1_ADM_s003
                                 + 1.40221 w_k1_AXP_s003
                                 + 1.66203 w_k1_DIS_s003
                                 + 1.40962 w_k1_FDX_s003
                                 + 1.50945 w_k1_IBM_s003
                                 + 1.40996 w_k1_INTC_s003
                                 + 1.55997 w_k1_KO_s003 + 1.32564 w_k1_MAT_s003
                                 + 1.34685 w_k1_MSFT_s003
                                 + 1.26362 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.20208 w_k1_ADM_s003
                                 + 1.04707 w_k1_AXP_s003 + 1.0442 w_k1_DIS_s003
                                 + 1.13705 w_k1_FDX_s003
                                 + 0.824386 w_k1_IBM_s003
                                 + 0.928976 w_k1_INTC_s003
                                 + 1.05267 w_k1_KO_s003
                                 + 0.835038 w_k1_MAT_s003
                                 + 1.03731 w_k1_MSFT_s003
                                 + 0.982708 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.02915 w_k1_ADM_s003
                                 + 1.10403 w_k1_AXP_s003
                                 + 1.34949 w_k1_DIS_s003
                                 + 1.25172 w_k1_FDX_s003
                                 + 0.987886 w_k1_IBM_s003
                                 + 1.1142 w_k1_INTC_s003 + 1.01109 w_k1_KO_s003
                                 + 0.737114 w_k1_MAT_s003
                                 + 1.23004 w_k1_MSFT_s003
                                 + 0.97915 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.994867 w_k1_ADM_s003
                                 + 0.697665 w_k1_AXP_s003
                                 + 0.683438 w_k1_DIS_s003
                                 + 0.798256 w_k1_FDX_s003
                                 + 0.709973 w_k1_IBM_s003
                                 + 0.79532 w_k1_INTC_s003
                                 + 0.667879 w_k1_KO_s003
                                 + 0.582092 w_k1_MAT_s003
                                 + 0.804332 w_k1_MSFT_s003
                                 + 0.823177 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.04099 w_k1_ADM_s003
                                 + 0.844597 w_k1_AXP_s003
                                 + 0.816849 w_k1_DIS_s003
                                 + 1.08708 w_k1_FDX_s003
                                 + 0.713888 w_k1_IBM_s003
                                 + 0.786557 w_k1_INTC_s003
                                 + 0.781571 w_k1_KO_s003
                                 + 0.682248 w_k1_MAT_s003
                                 + 0.896013 w_k1_MSFT_s003
                                 + 0.886844 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.06049 w_k1_ADM_s003
                                 + 0.867626 w_k1_AXP_s003
                                 + 0.849439 w_k1_DIS_s003
                                 + 0.973931 w_k1_FDX_s003
                                 + 0.916798 w_k1_IBM_s003
                                 + 0.88353 w_k1_INTC_s003
                                 + 0.847759 w_k1_KO_s003
                                 + 0.812198 w_k1_MAT_s003
                                 + 0.900991 w_k1_MSFT_s003
                                 + 0.905083 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.909823 w_k1_ADM_s003
                                 + 1.10814 w_k1_AXP_s003 + 1.1872 w_k1_DIS_s003
                                 + 1.00332 w_k1_FDX_s003
                                 + 1.36856 w_k1_IBM_s003
                                 + 0.950289 w_k1_INTC_s003
                                 + 1.26664 w_k1_KO_s003 + 1.27523 w_k1_MAT_s003
                                 + 1.01311 w_k1_MSFT_s003
                                 + 1.07337 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.71605 w_k1_ADM_s003
                                 + 0.704909 w_k1_AXP_s003
                                 + 0.890255 w_k1_DIS_s003
                                 + 0.815632 w_k1_FDX_s003
                                 + 1.21263 w_k1_IBM_s003
                                 + 0.721993 w_k1_INTC_s003
                                 + 1.02613 w_k1_KO_s003 + 0.95789 w_k1_MAT_s003
                                 + 0.847377 w_k1_MSFT_s003
                                 + 0.838854 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.996105 w_k1_ADM_s004
                                 + 0.852313 w_k1_AXP_s004
                                 + 0.862179 w_k1_DIS_s004
                                 + 0.963557 w_k1_FDX_s004
                                 + 0.737361 w_k1_IBM_s004
                                 + 1.01747 w_k1_INTC_s004
                                 + 0.859926 w_k1_KO_s004
                                 + 1.02987 w_k1_MAT_s004
                                 + 1.02208 w_k1_MSFT_s004
                                 + 0.947426 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.22934 w_k1_ADM_s004
                                 + 1.48427 w_k1_AXP_s004 + 1.6776 w_k1_DIS_s004
                                 + 1.43329 w_k1_FDX_s004
                                 + 1.13791 w_k1_IBM_s004
                                 + 1.22482 w_k1_INTC_s004 + 1.2487 w_k1_KO_s004
                                 + 1.23066 w_k1_MAT_s004
                                 + 1.19692 w_k1_MSFT_s004
                                 + 1.28895 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.780668 w_k1_ADM_s004
                                 + 0.748096 w_k1_AXP_s004
                                 + 0.829695 w_k1_DIS_s004
                                 + 0.840853 w_k1_FDX_s004
                                 + 1.03018 w_k1_IBM_s004
                                 + 0.899268 w_k1_INTC_s004
                                 + 0.929557 w_k1_KO_s004
                                 + 1.02637 w_k1_MAT_s004
                                 + 0.898523 w_k1_MSFT_s004
                                 + 0.92085 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.10472 w_k1_ADM_s004
                                 + 1.37627 w_k1_AXP_s004
                                 + 0.917086 w_k1_DIS_s004
                                 + 1.0659 w_k1_FDX_s004 + 1.10902 w_k1_IBM_s004
                                 + 0.982993 w_k1_INTC_s004
                                 + 0.934408 w_k1_KO_s004
                                 + 1.48004 w_k1_MAT_s004
                                 + 1.03459 w_k1_MSFT_s004
                                 + 1.03225 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.839829 w_k1_ADM_s004
                                 + 0.641052 w_k1_AXP_s004
                                 + 0.515881 w_k1_DIS_s004
                                 + 0.758998 w_k1_FDX_s004
                                 + 0.652041 w_k1_IBM_s004
                                 + 0.738521 w_k1_INTC_s004
                                 + 0.63755 w_k1_KO_s004
                                 + 0.649245 w_k1_MAT_s004
                                 + 0.743963 w_k1_MSFT_s004
                                 + 0.777533 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.0029 w_k1_ADM_s004
                                 + 1.2169 w_k1_AXP_s004 + 1.24612 w_k1_DIS_s004
                                 + 1.27508 w_k1_FDX_s004
                                 + 0.993662 w_k1_IBM_s004
                                 + 1.14337 w_k1_INTC_s004
                                 + 0.923626 w_k1_KO_s004
                                 + 0.970492 w_k1_MAT_s004
                                 + 1.26574 w_k1_MSFT_s004 + 1.1545 w_k1_PG_s004
                                 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.950543 w_k1_ADM_s004
                                 + 1.14172 w_k1_AXP_s004
                                 + 1.33737 w_k1_DIS_s004
                                 + 1.00435 w_k1_FDX_s004
                                 + 1.54935 w_k1_IBM_s004
                                 + 1.1092 w_k1_INTC_s004 + 1.16808 w_k1_KO_s004
                                 + 1.69261 w_k1_MAT_s004
                                 + 1.15216 w_k1_MSFT_s004
                                 + 1.06076 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.07373 w_k1_ADM_s004
                                 + 1.17917 w_k1_AXP_s004
                                 + 1.08993 w_k1_DIS_s004
                                 + 1.09633 w_k1_FDX_s004
                                 + 0.967616 w_k1_IBM_s004
                                 + 0.971074 w_k1_INTC_s004
                                 + 1.02614 w_k1_KO_s004 + 1.3559 w_k1_MAT_s004
                                 + 0.938478 w_k1_MSFT_s004
                                 + 1.18595 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.908592 w_k1_ADM_s005
                                 + 1.09884 w_k1_AXP_s005
                                 + 1.18875 w_k1_DIS_s005
                                 + 1.07925 w_k1_FDX_s005
                                 + 1.19892 w_k1_IBM_s005
                                 + 1.11834 w_k1_INTC_s005
                                 + 1.10609 w_k1_KO_s005
                                 + 0.870908 w_k1_MAT_s005
                                 + 1.09505 w_k1_MSFT_s005
                                 + 1.03828 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.789899 w_k1_ADM_s005
                                 + 1.20413 w_k1_AXP_s005
                                 + 1.06604 w_k1_DIS_s005
                                 + 0.962022 w_k1_FDX_s005
                                 + 1.6511 w_k1_IBM_s005
                                 + 0.832275 w_k1_INTC_s005
                                 + 1.31909 w_k1_KO_s005 + 1.23867 w_k1_MAT_s005
                                 + 0.989281 w_k1_MSFT_s005
                                 + 1.12686 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.0237 w_k1_ADM_s005
                                 + 1.03317 w_k1_AXP_s005
                                 + 0.921301 w_k1_DIS_s005
                                 + 1.03418 w_k1_FDX_s005
                                 + 0.668737 w_k1_IBM_s005
                                 + 1.04609 w_k1_INTC_s005
                                 + 0.768458 w_k1_KO_s005
                                 + 0.677696 w_k1_MAT_s005
                                 + 1.02218 w_k1_MSFT_s005
                                 + 0.943427 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.24624 w_k1_ADM_s005
                                 + 1.21442 w_k1_AXP_s005 + 1.3016 w_k1_DIS_s005
                                 + 1.35894 w_k1_FDX_s005
                                 + 0.714625 w_k1_IBM_s005
                                 + 1.42968 w_k1_INTC_s005
                                 + 0.89913 w_k1_KO_s005
                                 + 0.824847 w_k1_MAT_s005
                                 + 1.27098 w_k1_MSFT_s005
                                 + 1.16075 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.831619 w_k1_ADM_s005
                                 + 0.669757 w_k1_AXP_s005
                                 + 0.880811 w_k1_DIS_s005
                                 + 0.853416 w_k1_FDX_s005
                                 + 0.910118 w_k1_IBM_s005
                                 + 1.09445 w_k1_INTC_s005
                                 + 0.953225 w_k1_KO_s005
                                 + 0.647829 w_k1_MAT_s005
                                 + 0.963921 w_k1_MSFT_s005
                                 + 0.926432 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.996365 w_k1_ADM_s005
                                 + 0.69759 w_k1_AXP_s005
                                 + 0.731842 w_k1_DIS_s005
                                 + 0.874817 w_k1_FDX_s005
                                 + 0.601441 w_k1_IBM_s005
                                 + 0.693414 w_k1_INTC_s005
                                 + 0.633767 w_k1_KO_s005
                                 + 0.403509 w_k1_MAT_s005
                                 + 0.944624 w_k1_MSFT_s005
                                 + 0.747362 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.7769 w_k1_ADM_s005
                                 + 0.926748 w_k1_AXP_s005
                                 + 1.07379 w_k1_DIS_s005
                                 + 0.983944 w_k1_FDX_s005
                                 + 1.05431 w_k1_IBM_s005
                                 + 0.963977 w_k1_INTC_s005
                                 + 0.933963 w_k1_KO_s005
                                 + 1.06334 w_k1_MAT_s005
                                 + 1.09621 w_k1_MSFT_s005
                                 + 0.936605 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.11272 w_k1_ADM_s005
                                 + 1.65787 w_k1_AXP_s005 + 1.7764 w_k1_DIS_s005
                                 + 1.4175 w_k1_FDX_s005 + 1.31516 w_k1_IBM_s005
                                 + 1.41218 w_k1_INTC_s005
                                 + 1.37732 w_k1_KO_s005 + 1.38342 w_k1_MAT_s005
                                 + 1.43626 w_k1_MSFT_s005
                                 + 1.29181 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.04075 w_k1_ADM_s006
                                 + 0.992541 w_k1_AXP_s006
                                 + 1.06055 w_k1_DIS_s006 + 1.0262 w_k1_FDX_s006
                                 + 1.29258 w_k1_IBM_s006 + 1.05 w_k1_INTC_s006
                                 + 1.22673 w_k1_KO_s006
                                 + 0.907051 w_k1_MAT_s006
                                 + 0.977639 w_k1_MSFT_s006
                                 + 0.987803 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.660949 w_k1_ADM_s006
                                 + 0.677756 w_k1_AXP_s006
                                 + 0.544865 w_k1_DIS_s006
                                 + 0.754667 w_k1_FDX_s006
                                 + 0.766069 w_k1_IBM_s006
                                 + 0.781891 w_k1_INTC_s006
                                 + 0.703577 w_k1_KO_s006
                                 + 0.757656 w_k1_MAT_s006
                                 + 0.767626 w_k1_MSFT_s006
                                 + 0.765461 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.905703 w_k1_ADM_s006
                                 + 0.87838 w_k1_AXP_s006
                                 + 0.99278 w_k1_DIS_s006
                                 + 1.03295 w_k1_FDX_s006
                                 + 0.694626 w_k1_IBM_s006
                                 + 1.00756 w_k1_INTC_s006
                                 + 0.895382 w_k1_KO_s006
                                 + 0.868838 w_k1_MAT_s006
                                 + 1.0537 w_k1_MSFT_s006
                                 + 0.879109 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.599023 w_k1_ADM_s006
                                 + 0.829703 w_k1_AXP_s006
                                 + 0.783769 w_k1_DIS_s006
                                 + 0.784359 w_k1_FDX_s006
                                 + 1.24216 w_k1_IBM_s006
                                 + 1.00198 w_k1_INTC_s006
                                 + 1.06611 w_k1_KO_s006
                                 + 0.830039 w_k1_MAT_s006
                                 + 0.841705 w_k1_MSFT_s006
                                 + 0.95697 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.07366 w_k1_ADM_s006
                                 + 1.10538 w_k1_AXP_s006
                                 + 0.973936 w_k1_DIS_s006
                                 + 1.09647 w_k1_FDX_s006
                                 + 0.900182 w_k1_IBM_s006
                                 + 0.961166 w_k1_INTC_s006
                                 + 0.878688 w_k1_KO_s006
                                 + 0.695349 w_k1_MAT_s006
                                 + 1.01845 w_k1_MSFT_s006
                                 + 0.889539 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.909755 w_k1_ADM_s006
                                 + 0.855956 w_k1_AXP_s006
                                 + 0.846815 w_k1_DIS_s006
                                 + 0.939037 w_k1_FDX_s006
                                 + 0.959832 w_k1_IBM_s006
                                 + 0.947752 w_k1_INTC_s006
                                 + 0.85454 w_k1_KO_s006
                                 + 0.529755 w_k1_MAT_s006
                                 + 0.907827 w_k1_MSFT_s006
                                 + 0.874373 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.40038 w_k1_ADM_s006
                                 + 1.29544 w_k1_AXP_s006
                                 + 1.49768 w_k1_DIS_s006
                                 + 1.28824 w_k1_FDX_s006 + 1.205 w_k1_IBM_s006
                                 + 1.1577 w_k1_INTC_s006 + 1.29234 w_k1_KO_s006
                                 + 0.916046 w_k1_MAT_s006
                                 + 1.15204 w_k1_MSFT_s006 + 1.1307 w_k1_PG_s006
                                 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.745995 w_k1_ADM_s006
                                 + 0.634208 w_k1_AXP_s006
                                 + 0.727509 w_k1_DIS_s006
                                 + 0.872204 w_k1_FDX_s006
                                 + 0.722545 w_k1_IBM_s006
                                 + 0.764203 w_k1_INTC_s006
                                 + 0.828543 w_k1_KO_s006
                                 + 0.61798 w_k1_MAT_s006
                                 + 0.840301 w_k1_MSFT_s006
                                 + 0.79389 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.47221 w_k1_ADM_s007
                                 + 1.40392 w_k1_AXP_s007
                                 + 1.70052 w_k1_DIS_s007
                                 + 1.34678 w_k1_FDX_s007
                                 + 1.11802 w_k1_IBM_s007
                                 + 1.04745 w_k1_INTC_s007
                                 + 1.16921 w_k1_KO_s007 + 1.48342 w_k1_MAT_s007
                                 + 1.311 w_k1_MSFT_s007 + 1.02287 w_k1_PG_s007
                                 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.896806 w_k1_ADM_s007
                                 + 1.0636 w_k1_AXP_s007
                                 + 0.698221 w_k1_DIS_s007
                                 + 0.753831 w_k1_FDX_s007
                                 + 1.43208 w_k1_IBM_s007
                                 + 0.673671 w_k1_INTC_s007
                                 + 0.944091 w_k1_KO_s007 + 2.2358 w_k1_MAT_s007
                                 + 0.804218 w_k1_MSFT_s007
                                 + 0.840424 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.972451 w_k1_ADM_s007
                                 + 0.996723 w_k1_AXP_s007 + 1.109 w_k1_DIS_s007
                                 + 1.17 w_k1_FDX_s007 + 0.651219 w_k1_IBM_s007
                                 + 1.02027 w_k1_INTC_s007
                                 + 0.96253 w_k1_KO_s007
                                 + 0.764725 w_k1_MAT_s007
                                 + 1.25671 w_k1_MSFT_s007
                                 + 0.989806 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.927808 w_k1_ADM_s007
                                 + 1.22334 w_k1_AXP_s007
                                 + 0.849666 w_k1_DIS_s007
                                 + 1.01672 w_k1_FDX_s007
                                 + 0.865059 w_k1_IBM_s007
                                 + 0.812922 w_k1_INTC_s007
                                 + 0.883826 w_k1_KO_s007
                                 + 1.47571 w_k1_MAT_s007
                                 + 0.8591 w_k1_MSFT_s007
                                 + 0.881221 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.10342 w_k1_ADM_s007
                                 + 1.74925 w_k1_AXP_s007 + 1.5357 w_k1_DIS_s007
                                 + 1.25337 w_k1_FDX_s007
                                 + 1.58789 w_k1_IBM_s007
                                 + 1.20706 w_k1_INTC_s007
                                 + 1.46145 w_k1_KO_s007 + 2.63309 w_k1_MAT_s007
                                 + 1.26268 w_k1_MSFT_s007
                                 + 1.10191 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.11648 w_k1_ADM_s007
                                 + 1.1867 w_k1_AXP_s007 + 1.20333 w_k1_DIS_s007
                                 + 1.05701 w_k1_FDX_s007
                                 + 1.03816 w_k1_IBM_s007
                                 + 1.09587 w_k1_INTC_s007
                                 + 1.16852 w_k1_KO_s007 + 1.36353 w_k1_MAT_s007
                                 + 1.1413 w_k1_MSFT_s007 + 1.0918 w_k1_PG_s007
                                 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.36321 w_k1_ADM_s007
                                 + 1.05849 w_k1_AXP_s007
                                 + 1.02367 w_k1_DIS_s007
                                 + 1.01058 w_k1_FDX_s007 + 1.165 w_k1_IBM_s007
                                 + 0.916711 w_k1_INTC_s007
                                 + 1.04875 w_k1_KO_s007 + 1.29282 w_k1_MAT_s007
                                 + 0.886171 w_k1_MSFT_s007
                                 + 1.00574 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.918612 w_k1_ADM_s007
                                 + 0.693425 w_k1_AXP_s007
                                 + 0.748594 w_k1_DIS_s007
                                 + 0.78129 w_k1_FDX_s007
                                 + 0.950721 w_k1_IBM_s007
                                 + 0.745866 w_k1_INTC_s007
                                 + 0.936525 w_k1_KO_s007
                                 + 1.12275 w_k1_MAT_s007
                                 + 0.916166 w_k1_MSFT_s007
                                 + 0.912949 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.781186 w_k1_ADM_s010
                                 + 1.09634 w_k1_AXP_s010
                                 + 0.908204 w_k1_DIS_s010
                                 + 0.840327 w_k1_FDX_s010
                                 + 1.5867 w_k1_IBM_s010
                                 + 0.935229 w_k1_INTC_s010
                                 + 1.04825 w_k1_KO_s010 + 1.24416 w_k1_MAT_s010
                                 + 0.768705 w_k1_MSFT_s010
                                 + 0.950785 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.09725 w_k1_ADM_s010
                                 + 1.07194 w_k1_AXP_s010
                                 + 1.46636 w_k1_DIS_s010
                                 + 1.26759 w_k1_FDX_s010
                                 + 0.818483 w_k1_IBM_s010
                                 + 1.24491 w_k1_INTC_s010
                                 + 1.34048 w_k1_KO_s010
                                 + 0.761628 w_k1_MAT_s010
                                 + 1.20559 w_k1_MSFT_s010
                                 + 1.29695 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.872436 w_k1_ADM_s010
                                 + 0.851772 w_k1_AXP_s010
                                 + 0.895351 w_k1_DIS_s010
                                 + 1.00393 w_k1_FDX_s010
                                 + 0.803491 w_k1_IBM_s010
                                 + 1.21434 w_k1_INTC_s010
                                 + 0.83448 w_k1_KO_s010
                                 + 0.678825 w_k1_MAT_s010
                                 + 1.08114 w_k1_MSFT_s010
                                 + 0.943589 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.679726 w_k1_ADM_s010
                                 + 0.62135 w_k1_AXP_s010
                                 + 0.597078 w_k1_DIS_s010
                                 + 0.734228 w_k1_FDX_s010
                                 + 0.710945 w_k1_IBM_s010
                                 + 0.907224 w_k1_INTC_s010
                                 + 0.817219 w_k1_KO_s010
                                 + 0.644851 w_k1_MAT_s010
                                 + 0.730928 w_k1_MSFT_s010
                                 + 0.890794 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.766066 w_k1_ADM_s010
                                 + 0.988834 w_k1_AXP_s010
                                 + 0.949357 w_k1_DIS_s010
                                 + 1.14608 w_k1_FDX_s010
                                 + 0.607733 w_k1_IBM_s010
                                 + 1.30108 w_k1_INTC_s010
                                 + 0.764878 w_k1_KO_s010
                                 + 0.572365 w_k1_MAT_s010
                                 + 1.2058 w_k1_MSFT_s010 + 1.06414 w_k1_PG_s010
                                 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.04935 w_k1_ADM_s010
                                 + 1.47724 w_k1_AXP_s010
                                 + 1.37016 w_k1_DIS_s010
                                 + 1.22301 w_k1_FDX_s010
                                 + 1.30559 w_k1_IBM_s010
                                 + 1.24632 w_k1_INTC_s010
                                 + 1.28603 w_k1_KO_s010 + 1.37143 w_k1_MAT_s010
                                 + 1.14559 w_k1_MSFT_s010
                                 + 1.16399 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.789122 w_k1_ADM_s010
                                 + 1.11119 w_k1_AXP_s010
                                 + 0.941315 w_k1_DIS_s010
                                 + 0.947866 w_k1_FDX_s010
                                 + 1.04655 w_k1_IBM_s010
                                 + 0.997217 w_k1_INTC_s010
                                 + 1.03972 w_k1_KO_s010 + 1.0009 w_k1_MAT_s010
                                 + 0.874417 w_k1_MSFT_s010
                                 + 0.903541 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.0944 w_k1_ADM_s010
                                 + 1.97174 w_k1_AXP_s010
                                 + 1.80714 w_k1_DIS_s010
                                 + 1.34276 w_k1_FDX_s010
                                 + 1.79117 w_k1_IBM_s010
                                 + 1.42775 w_k1_INTC_s010
                                 + 1.50591 w_k1_KO_s010 + 1.72038 w_k1_MAT_s010
                                 + 1.13843 w_k1_MSFT_s010
                                 + 1.40461 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.14534 w_k1_ADM_s011
                                 + 1.41092 w_k1_AXP_s011
                                 + 1.27341 w_k1_DIS_s011
                                 + 1.26324 w_k1_FDX_s011
                                 + 1.06671 w_k1_IBM_s011
                                 + 1.06596 w_k1_INTC_s011
                                 + 1.18316 w_k1_KO_s011 + 1.36464 w_k1_MAT_s011
                                 + 1.1489 w_k1_MSFT_s011 + 1.12239 w_k1_PG_s011
                                 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.835968 w_k1_ADM_s011
                                 + 1.35108 w_k1_AXP_s011
                                 + 0.914296 w_k1_DIS_s011
                                 + 1.04955 w_k1_FDX_s011
                                 + 1.08006 w_k1_IBM_s011
                                 + 0.998091 w_k1_INTC_s011
                                 + 1.02695 w_k1_KO_s011 + 1.40211 w_k1_MAT_s011
                                 + 1.02222 w_k1_MSFT_s011
                                 + 0.945481 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.61718 w_k1_ADM_s011
                                 + 0.640391 w_k1_AXP_s011
                                 + 0.713285 w_k1_DIS_s011
                                 + 0.813563 w_k1_FDX_s011
                                 + 0.748271 w_k1_IBM_s011
                                 + 0.818423 w_k1_INTC_s011
                                 + 0.844198 w_k1_KO_s011
                                 + 0.558478 w_k1_MAT_s011
                                 + 0.838468 w_k1_MSFT_s011
                                 + 0.847858 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.95293 w_k1_ADM_s011
                                 + 1.22365 w_k1_AXP_s011
                                 + 1.45358 w_k1_DIS_s011 + 1.2345 w_k1_FDX_s011
                                 + 1.51233 w_k1_IBM_s011
                                 + 1.15139 w_k1_INTC_s011
                                 + 1.56077 w_k1_KO_s011 + 1.51862 w_k1_MAT_s011
                                 + 1.12378 w_k1_MSFT_s011
                                 + 1.27361 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.940749 w_k1_ADM_s011
                                 + 1.02947 w_k1_AXP_s011
                                 + 0.931603 w_k1_DIS_s011
                                 + 1.03665 w_k1_FDX_s011
                                 + 0.955597 w_k1_IBM_s011
                                 + 1.0639 w_k1_INTC_s011 + 1.03415 w_k1_KO_s011
                                 + 0.858961 w_k1_MAT_s011
                                 + 0.97175 w_k1_MSFT_s011
                                 + 1.00522 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.974472 w_k1_ADM_s011
                                 + 0.89609 w_k1_AXP_s011
                                 + 1.04951 w_k1_DIS_s011
                                 + 1.01884 w_k1_FDX_s011 + 1.2922 w_k1_IBM_s011
                                 + 1.14317 w_k1_INTC_s011
                                 + 1.42798 w_k1_KO_s011 + 1.38869 w_k1_MAT_s011
                                 + 1.10903 w_k1_MSFT_s011
                                 + 1.17931 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.648891 w_k1_ADM_s011
                                 + 0.896494 w_k1_AXP_s011
                                 + 0.740322 w_k1_DIS_s011
                                 + 0.791402 w_k1_FDX_s011
                                 + 1.34969 w_k1_IBM_s011
                                 + 0.961082 w_k1_INTC_s011
                                 + 1.14324 w_k1_KO_s011 + 1.33525 w_k1_MAT_s011
                                 + 0.785706 w_k1_MSFT_s011
                                 + 0.936694 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.880208 w_k1_ADM_s011
                                 + 0.93937 w_k1_AXP_s011
                                 + 1.15103 w_k1_DIS_s011
                                 + 1.04804 w_k1_FDX_s011
                                 + 1.12944 w_k1_IBM_s011
                                 + 0.96445 w_k1_INTC_s011
                                 + 1.13372 w_k1_KO_s011 + 1.27106 w_k1_MAT_s011
                                 + 1.13686 w_k1_MSFT_s011
                                 + 1.04885 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.16054 w_k1_ADM_s012
                                 + 1.28276 w_k1_AXP_s012
                                 + 1.00078 w_k1_DIS_s012
                                 + 1.14374 w_k1_FDX_s012 + 1.472 w_k1_IBM_s012
                                 + 1.18256 w_k1_INTC_s012
                                 + 1.07934 w_k1_KO_s012 + 1.34777 w_k1_MAT_s012
                                 + 1.06855 w_k1_MSFT_s012
                                 + 1.09324 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.1562 w_k1_ADM_s012
                                 + 1.22885 w_k1_AXP_s012
                                 + 0.987264 w_k1_DIS_s012
                                 + 1.12022 w_k1_FDX_s012 + 0.8682 w_k1_IBM_s012
                                 + 1.06174 w_k1_INTC_s012
                                 + 0.866549 w_k1_KO_s012
                                 + 1.02887 w_k1_MAT_s012
                                 + 1.06842 w_k1_MSFT_s012
                                 + 0.928377 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.19974 w_k1_ADM_s012
                                 + 0.896919 w_k1_AXP_s012
                                 + 1.0465 w_k1_DIS_s012 + 1.03413 w_k1_FDX_s012
                                 + 0.736405 w_k1_IBM_s012
                                 + 1.15658 w_k1_INTC_s012
                                 + 0.876791 w_k1_KO_s012
                                 + 0.663333 w_k1_MAT_s012
                                 + 1.19742 w_k1_MSFT_s012
                                 + 0.91116 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.71979 w_k1_ADM_s012
                                 + 0.819617 w_k1_AXP_s012
                                 + 0.779388 w_k1_DIS_s012
                                 + 0.766588 w_k1_FDX_s012
                                 + 1.0224 w_k1_IBM_s012
                                 + 0.938879 w_k1_INTC_s012
                                 + 0.845506 w_k1_KO_s012
                                 + 0.955546 w_k1_MAT_s012
                                 + 0.914315 w_k1_MSFT_s012
                                 + 0.907078 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.875618 w_k1_ADM_s012
                                 + 1.1432 w_k1_AXP_s012 + 1.06842 w_k1_DIS_s012
                                 + 0.979607 w_k1_FDX_s012
                                 + 1.50434 w_k1_IBM_s012
                                 + 1.14607 w_k1_INTC_s012
                                 + 1.13467 w_k1_KO_s012 + 1.3312 w_k1_MAT_s012
                                 + 1.16611 w_k1_MSFT_s012
                                 + 1.16106 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.991204 w_k1_ADM_s012
                                 + 1.0306 w_k1_AXP_s012
                                 + 0.907016 w_k1_DIS_s012
                                 + 0.928851 w_k1_FDX_s012
                                 + 1.09944 w_k1_IBM_s012
                                 + 0.924412 w_k1_INTC_s012
                                 + 0.915484 w_k1_KO_s012
                                 + 0.788294 w_k1_MAT_s012
                                 + 1.03904 w_k1_MSFT_s012
                                 + 0.926428 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.418 w_k1_ADM_s012
                                 + 1.5447 w_k1_AXP_s012 + 1.6391 w_k1_DIS_s012
                                 + 1.36017 w_k1_FDX_s012
                                 + 1.16244 w_k1_IBM_s012
                                 + 1.37467 w_k1_INTC_s012
                                 + 1.18608 w_k1_KO_s012 + 1.2887 w_k1_MAT_s012
                                 + 1.65293 w_k1_MSFT_s012
                                 + 1.19811 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.803667 w_k1_ADM_s012
                                 + 0.694892 w_k1_AXP_s012
                                 + 0.711437 w_k1_DIS_s012
                                 + 0.764688 w_k1_FDX_s012
                                 + 0.67492 w_k1_IBM_s012
                                 + 0.810392 w_k1_INTC_s012
                                 + 0.774361 w_k1_KO_s012
                                 + 0.575269 w_k1_MAT_s012
                                 + 0.856256 w_k1_MSFT_s012
                                 + 0.846565 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.06287 w_k1_ADM_s013
                                 + 1.10923 w_k1_AXP_s013 + 1.0329 w_k1_DIS_s013
                                 + 0.932831 w_k1_FDX_s013
                                 + 1.2067 w_k1_IBM_s013
                                 + 0.941891 w_k1_INTC_s013
                                 + 1.06505 w_k1_KO_s013 + 1.50067 w_k1_MAT_s013
                                 + 0.831791 w_k1_MSFT_s013
                                 + 1.17812 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.778012 w_k1_ADM_s013
                                 + 0.77743 w_k1_AXP_s013
                                 + 0.794509 w_k1_DIS_s013
                                 + 0.78508 w_k1_FDX_s013
                                 + 0.787851 w_k1_IBM_s013
                                 + 0.846602 w_k1_INTC_s013
                                 + 0.826662 w_k1_KO_s013
                                 + 0.821123 w_k1_MAT_s013
                                 + 0.738533 w_k1_MSFT_s013
                                 + 0.864815 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.32536 w_k1_ADM_s013
                                 + 0.974901 w_k1_AXP_s013
                                 + 1.47361 w_k1_DIS_s013
                                 + 1.25778 w_k1_FDX_s013
                                 + 0.854969 w_k1_IBM_s013
                                 + 1.05986 w_k1_INTC_s013
                                 + 1.07392 w_k1_KO_s013
                                 + 0.832952 w_k1_MAT_s013
                                 + 1.07089 w_k1_MSFT_s013
                                 + 1.18111 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.28601 w_k1_ADM_s013
                                 + 1.41297 w_k1_AXP_s013
                                 + 1.11942 w_k1_DIS_s013
                                 + 1.14785 w_k1_FDX_s013
                                 + 0.733786 w_k1_IBM_s013
                                 + 1.19795 w_k1_INTC_s013
                                 + 0.955089 w_k1_KO_s013 + 1.2599 w_k1_MAT_s013
                                 + 1.03923 w_k1_MSFT_s013
                                 + 1.04464 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.01426 w_k1_ADM_s013
                                 + 1.16287 w_k1_AXP_s013
                                 + 0.894842 w_k1_DIS_s013
                                 + 1.07087 w_k1_FDX_s013
                                 + 0.939981 w_k1_IBM_s013
                                 + 0.972729 w_k1_INTC_s013
                                 + 0.898514 w_k1_KO_s013
                                 + 1.39217 w_k1_MAT_s013
                                 + 0.939233 w_k1_MSFT_s013
                                 + 1.11618 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.901636 w_k1_ADM_s013
                                 + 0.976112 w_k1_AXP_s013
                                 + 1.09103 w_k1_DIS_s013
                                 + 0.993738 w_k1_FDX_s013
                                 + 0.732054 w_k1_IBM_s013
                                 + 1.19038 w_k1_INTC_s013
                                 + 0.875195 w_k1_KO_s013
                                 + 0.770011 w_k1_MAT_s013
                                 + 1.09703 w_k1_MSFT_s013
                                 + 0.966119 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.20216 w_k1_ADM_s013
                                 + 0.987112 w_k1_AXP_s013
                                 + 0.929581 w_k1_DIS_s013
                                 + 1.04432 w_k1_FDX_s013
                                 + 0.619432 w_k1_IBM_s013
                                 + 0.834351 w_k1_INTC_s013
                                 + 0.756261 w_k1_KO_s013
                                 + 0.61897 w_k1_MAT_s013
                                 + 0.802038 w_k1_MSFT_s013
                                 + 0.973685 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.0844 w_k1_ADM_s013
                                 + 1.53934 w_k1_AXP_s013
                                 + 1.46874 w_k1_DIS_s013
                                 + 1.27631 w_k1_FDX_s013
                                 + 1.31465 w_k1_IBM_s013
                                 + 1.24214 w_k1_INTC_s013
                                 + 1.09012 w_k1_KO_s013 + 1.28358 w_k1_MAT_s013
                                 + 1.01397 w_k1_MSFT_s013
                                 + 1.20094 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.48764 w_k1_ADM_s014
                                 + 1.73122 w_k1_AXP_s014
                                 + 1.55009 w_k1_DIS_s014
                                 + 1.52033 w_k1_FDX_s014
                                 + 0.990554 w_k1_IBM_s014
                                 + 1.17115 w_k1_INTC_s014
                                 + 1.05983 w_k1_KO_s014 + 1.59476 w_k1_MAT_s014
                                 + 1.24167 w_k1_MSFT_s014
                                 + 1.19498 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.15029 w_k1_ADM_s014
                                 + 0.984815 w_k1_AXP_s014
                                 + 0.943993 w_k1_DIS_s014
                                 + 1.05118 w_k1_FDX_s014
                                 + 0.773163 w_k1_IBM_s014
                                 + 1.08828 w_k1_INTC_s014
                                 + 0.843964 w_k1_KO_s014
                                 + 0.750348 w_k1_MAT_s014
                                 + 1.04164 w_k1_MSFT_s014
                                 + 0.973424 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.45637 w_k1_ADM_s014
                                 + 1.10003 w_k1_AXP_s014
                                 + 1.33391 w_k1_DIS_s014
                                 + 1.19738 w_k1_FDX_s014
                                 + 0.896943 w_k1_IBM_s014
                                 + 1.30895 w_k1_INTC_s014
                                 + 1.13076 w_k1_KO_s014
                                 + 0.822669 w_k1_MAT_s014
                                 + 1.26629 w_k1_MSFT_s014
                                 + 1.18149 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.983128 w_k1_ADM_s014
                                 + 0.755352 w_k1_AXP_s014
                                 + 0.877109 w_k1_DIS_s014
                                 + 0.943285 w_k1_FDX_s014
                                 + 1.10168 w_k1_IBM_s014
                                 + 0.82424 w_k1_INTC_s014
                                 + 1.05143 w_k1_KO_s014 + 1.20129 w_k1_MAT_s014
                                 + 0.975352 w_k1_MSFT_s014
                                 + 1.04548 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.52796 w_k1_ADM_s014
                                 + 1.41667 w_k1_AXP_s014
                                 + 1.38955 w_k1_DIS_s014
                                 + 1.15243 w_k1_FDX_s014
                                 + 1.50748 w_k1_IBM_s014
                                 + 1.35267 w_k1_INTC_s014
                                 + 1.36654 w_k1_KO_s014 + 1.73768 w_k1_MAT_s014
                                 + 1.25177 w_k1_MSFT_s014
                                 + 1.21716 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.987214 w_k1_ADM_s014
                                 + 0.762889 w_k1_AXP_s014
                                 + 0.654354 w_k1_DIS_s014
                                 + 0.679795 w_k1_FDX_s014
                                 + 0.905878 w_k1_IBM_s014
                                 + 0.836812 w_k1_INTC_s014
                                 + 0.834988 w_k1_KO_s014
                                 + 0.689605 w_k1_MAT_s014
                                 + 0.742632 w_k1_MSFT_s014
                                 + 0.885481 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.11595 w_k1_ADM_s014
                                 + 1.16113 w_k1_AXP_s014
                                 + 0.967073 w_k1_DIS_s014
                                 + 0.889002 w_k1_FDX_s014
                                 + 1.35862 w_k1_IBM_s014
                                 + 0.764248 w_k1_INTC_s014
                                 + 1.10934 w_k1_KO_s014 + 1.76268 w_k1_MAT_s014
                                 + 0.899115 w_k1_MSFT_s014
                                 + 1.14814 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.30345 w_k1_ADM_s014
                                 + 0.976157 w_k1_AXP_s014
                                 + 0.999779 w_k1_DIS_s014
                                 + 0.971954 w_k1_FDX_s014
                                 + 0.940511 w_k1_IBM_s014
                                 + 0.958546 w_k1_INTC_s014
                                 + 0.888475 w_k1_KO_s014
                                 + 1.20678 w_k1_MAT_s014
                                 + 0.985312 w_k1_MSFT_s014
                                 + 0.946526 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.05075 w_k1_ADM_s015
                                 + 1.05145 w_k1_AXP_s015
                                 + 1.07654 w_k1_DIS_s015
                                 + 1.05844 w_k1_FDX_s015
                                 + 1.02116 w_k1_IBM_s015
                                 + 0.961731 w_k1_INTC_s015
                                 + 0.962505 w_k1_KO_s015
                                 + 0.979299 w_k1_MAT_s015
                                 + 0.905885 w_k1_MSFT_s015
                                 + 1.06045 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.903996 w_k1_ADM_s015
                                 + 0.653333 w_k1_AXP_s015
                                 + 0.527088 w_k1_DIS_s015
                                 + 0.703611 w_k1_FDX_s015
                                 + 0.718058 w_k1_IBM_s015
                                 + 0.59631 w_k1_INTC_s015
                                 + 0.631747 w_k1_KO_s015
                                 + 0.696015 w_k1_MAT_s015
                                 + 0.686418 w_k1_MSFT_s015
                                 + 0.76982 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.13542 w_k1_ADM_s015
                                 + 1.16283 w_k1_AXP_s015
                                 + 1.25807 w_k1_DIS_s015 + 1.2115 w_k1_FDX_s015
                                 + 0.791013 w_k1_IBM_s015
                                 + 0.877664 w_k1_INTC_s015
                                 + 1.00508 w_k1_KO_s015 + 0.88086 w_k1_MAT_s015
                                 + 1.07312 w_k1_MSFT_s015
                                 + 1.07191 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.21018 w_k1_ADM_s015
                                 + 0.765321 w_k1_AXP_s015
                                 + 0.835063 w_k1_DIS_s015
                                 + 0.896399 w_k1_FDX_s015
                                 + 0.809843 w_k1_IBM_s015
                                 + 0.7865 w_k1_INTC_s015
                                 + 0.832257 w_k1_KO_s015
                                 + 0.556972 w_k1_MAT_s015
                                 + 0.785221 w_k1_MSFT_s015
                                 + 0.919962 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.45276 w_k1_ADM_s015
                                 + 1.39162 w_k1_AXP_s015
                                 + 1.28997 w_k1_DIS_s015
                                 + 1.11021 w_k1_FDX_s015 + 1.711 w_k1_IBM_s015
                                 + 0.966745 w_k1_INTC_s015
                                 + 1.4297 w_k1_KO_s015 + 2.095 w_k1_MAT_s015
                                 + 1.08325 w_k1_MSFT_s015
                                 + 1.35471 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.02665 w_k1_ADM_s015
                                 + 0.784201 w_k1_AXP_s015
                                 + 0.643395 w_k1_DIS_s015
                                 + 0.839112 w_k1_FDX_s015
                                 + 0.571638 w_k1_IBM_s015
                                 + 0.793168 w_k1_INTC_s015
                                 + 0.679081 w_k1_KO_s015
                                 + 0.581543 w_k1_MAT_s015
                                 + 0.875615 w_k1_MSFT_s015
                                 + 0.883988 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.28483 w_k1_ADM_s015
                                 + 1.32937 w_k1_AXP_s015
                                 + 1.39222 w_k1_DIS_s015
                                 + 1.32543 w_k1_FDX_s015
                                 + 1.27289 w_k1_IBM_s015
                                 + 1.15262 w_k1_INTC_s015
                                 + 1.26294 w_k1_KO_s015 + 1.3851 w_k1_MAT_s015
                                 + 1.10172 w_k1_MSFT_s015
                                 + 1.15523 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.998397 w_k1_ADM_s015
                                 + 0.941287 w_k1_AXP_s015
                                 + 0.929406 w_k1_DIS_s015
                                 + 0.850748 w_k1_FDX_s015
                                 + 1.15797 w_k1_IBM_s015
                                 + 0.910595 w_k1_INTC_s015
                                 + 0.946214 w_k1_KO_s015
                                 + 1.01042 w_k1_MAT_s015
                                 + 0.843088 w_k1_MSFT_s015
                                 + 0.918977 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.17227 w_k1_ADM_s016
                                 + 0.902739 w_k1_AXP_s016
                                 + 1.23042 w_k1_DIS_s016
                                 + 1.14123 w_k1_FDX_s016
                                 + 0.823276 w_k1_IBM_s016
                                 + 1.06719 w_k1_INTC_s016
                                 + 0.973865 w_k1_KO_s016
                                 + 0.857366 w_k1_MAT_s016
                                 + 1.19256 w_k1_MSFT_s016
                                 + 0.937049 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.13138 w_k1_ADM_s016
                                 + 1.00314 w_k1_AXP_s016
                                 + 1.63802 w_k1_DIS_s016
                                 + 1.14045 w_k1_FDX_s016 + 1.1484 w_k1_IBM_s016
                                 + 1.20753 w_k1_INTC_s016 + 1.2304 w_k1_KO_s016
                                 + 0.96987 w_k1_MAT_s016
                                 + 1.14723 w_k1_MSFT_s016
                                 + 1.04279 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.689567 w_k1_ADM_s016
                                 + 0.785852 w_k1_AXP_s016
                                 + 0.679188 w_k1_DIS_s016
                                 + 0.796868 w_k1_FDX_s016
                                 + 0.728475 w_k1_IBM_s016
                                 + 0.992516 w_k1_INTC_s016
                                 + 0.668838 w_k1_KO_s016
                                 + 0.876729 w_k1_MAT_s016
                                 + 0.866543 w_k1_MSFT_s016
                                 + 0.775605 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.818968 w_k1_ADM_s016
                                 + 0.839022 w_k1_AXP_s016
                                 + 0.992901 w_k1_DIS_s016
                                 + 0.960846 w_k1_FDX_s016
                                 + 0.777968 w_k1_IBM_s016
                                 + 1.12685 w_k1_INTC_s016
                                 + 0.869864 w_k1_KO_s016
                                 + 0.866703 w_k1_MAT_s016
                                 + 0.958272 w_k1_MSFT_s016
                                 + 0.841769 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.952046 w_k1_ADM_s016
                                 + 0.881513 w_k1_AXP_s016
                                 + 1.03107 w_k1_DIS_s016
                                 + 0.877386 w_k1_FDX_s016
                                 + 1.25187 w_k1_IBM_s016
                                 + 1.10229 w_k1_INTC_s016
                                 + 1.15959 w_k1_KO_s016
                                 + 0.995942 w_k1_MAT_s016
                                 + 0.996209 w_k1_MSFT_s016
                                 + 0.977855 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.623083 w_k1_ADM_s016
                                 + 0.637572 w_k1_AXP_s016
                                 + 0.621527 w_k1_DIS_s016
                                 + 0.66107 w_k1_FDX_s016
                                 + 0.800961 w_k1_IBM_s016
                                 + 0.774798 w_k1_INTC_s016
                                 + 0.705189 w_k1_KO_s016
                                 + 0.951335 w_k1_MAT_s016
                                 + 0.61315 w_k1_MSFT_s016
                                 + 0.738707 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.17567 w_k1_ADM_s016
                                 + 1.35336 w_k1_AXP_s016
                                 + 1.54814 w_k1_DIS_s016
                                 + 1.36525 w_k1_FDX_s016
                                 + 1.23162 w_k1_IBM_s016
                                 + 1.35856 w_k1_INTC_s016
                                 + 1.32703 w_k1_KO_s016 + 1.69559 w_k1_MAT_s016
                                 + 1.17342 w_k1_MSFT_s016
                                 + 1.20618 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.984682 w_k1_ADM_s016
                                 + 0.980338 w_k1_AXP_s016
                                 + 1.11181 w_k1_DIS_s016
                                 + 1.05525 w_k1_FDX_s016
                                 + 1.02185 w_k1_IBM_s016
                                 + 1.04724 w_k1_INTC_s016
                                 + 0.911324 w_k1_KO_s016
                                 + 1.34952 w_k1_MAT_s016
                                 + 1.02193 w_k1_MSFT_s016
                                 + 0.997153 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.95792 w_k1_ADM_s017
                                 + 0.897165 w_k1_AXP_s017
                                 + 0.640253 w_k1_DIS_s017
                                 + 0.807503 w_k1_FDX_s017
                                 + 0.751578 w_k1_IBM_s017
                                 + 0.848645 w_k1_INTC_s017
                                 + 0.726753 w_k1_KO_s017
                                 + 0.663921 w_k1_MAT_s017
                                 + 0.87896 w_k1_MSFT_s017
                                 + 0.873551 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.71841 w_k1_ADM_s017
                                 + 1.53605 w_k1_AXP_s017
                                 + 1.65962 w_k1_DIS_s017
                                 + 1.36498 w_k1_FDX_s017
                                 + 1.55322 w_k1_IBM_s017
                                 + 1.59689 w_k1_INTC_s017
                                 + 1.48842 w_k1_KO_s017 + 1.3429 w_k1_MAT_s017
                                 + 1.56092 w_k1_MSFT_s017
                                 + 1.23936 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.12282 w_k1_ADM_s017
                                 + 0.920793 w_k1_AXP_s017
                                 + 1.19402 w_k1_DIS_s017 + 1.0512 w_k1_FDX_s017
                                 + 0.933741 w_k1_IBM_s017
                                 + 0.920664 w_k1_INTC_s017
                                 + 1.0595 w_k1_KO_s017 + 0.931541 w_k1_MAT_s017
                                 + 1.17291 w_k1_MSFT_s017 + 1.0856 w_k1_PG_s017
                                 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.08601 w_k1_ADM_s017
                                 + 1.00689 w_k1_AXP_s017
                                 + 1.34378 w_k1_DIS_s017
                                 + 1.09903 w_k1_FDX_s017
                                 + 1.49057 w_k1_IBM_s017
                                 + 1.05077 w_k1_INTC_s017
                                 + 1.20469 w_k1_KO_s017 + 1.57784 w_k1_MAT_s017
                                 + 1.40901 w_k1_MSFT_s017 + 1.2187 w_k1_PG_s017
                                 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.29449 w_k1_ADM_s017
                                 + 0.994432 w_k1_AXP_s017
                                 + 0.974139 w_k1_DIS_s017
                                 + 1.06227 w_k1_FDX_s017
                                 + 0.772891 w_k1_IBM_s017
                                 + 0.975416 w_k1_INTC_s017
                                 + 0.901157 w_k1_KO_s017
                                 + 0.690127 w_k1_MAT_s017
                                 + 1.12298 w_k1_MSFT_s017
                                 + 1.06603 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.19319 w_k1_ADM_s017
                                 + 1.10951 w_k1_AXP_s017
                                 + 0.866922 w_k1_DIS_s017
                                 + 0.991793 w_k1_FDX_s017
                                 + 1.1607 w_k1_IBM_s017
                                 + 1.06309 w_k1_INTC_s017
                                 + 0.999608 w_k1_KO_s017
                                 + 1.28504 w_k1_MAT_s017
                                 + 1.1789 w_k1_MSFT_s017 + 1.06262 w_k1_PG_s017
                                 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.50495 w_k1_ADM_s017
                                 + 1.26504 w_k1_AXP_s017
                                 + 1.32548 w_k1_DIS_s017
                                 + 1.30936 w_k1_FDX_s017
                                 + 1.03197 w_k1_IBM_s017
                                 + 1.11606 w_k1_INTC_s017
                                 + 1.02768 w_k1_KO_s017
                                 + 0.929902 w_k1_MAT_s017
                                 + 1.24335 w_k1_MSFT_s017
                                 + 1.11248 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.977342 w_k1_ADM_s017
                                 + 0.84401 w_k1_AXP_s017
                                 + 0.791688 w_k1_DIS_s017
                                 + 0.902738 w_k1_FDX_s017
                                 + 0.972526 w_k1_IBM_s017
                                 + 0.945801 w_k1_INTC_s017
                                 + 0.83944 w_k1_KO_s017 + 0.91731 w_k1_MAT_s017
                                 + 1.05194 w_k1_MSFT_s017
                                 + 1.10152 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.42681 w_k1_ADM_s100
                                 + 1.67091 w_k1_AXP_s100
                                 + 1.93219 w_k1_DIS_s100 + 1.4403 w_k1_FDX_s100
                                 + 1.31495 w_k1_IBM_s100
                                 + 1.29907 w_k1_INTC_s100
                                 + 1.37356 w_k1_KO_s100 + 1.5745 w_k1_MAT_s100
                                 + 1.34009 w_k1_MSFT_s100
                                 + 1.31668 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.05106 w_k1_ADM_s100
                                 + 0.81676 w_k1_AXP_s100
                                 + 0.855121 w_k1_DIS_s100
                                 + 0.911213 w_k1_FDX_s100
                                 + 1.15059 w_k1_IBM_s100
                                 + 0.717752 w_k1_INTC_s100
                                 + 0.974427 w_k1_KO_s100
                                 + 1.55993 w_k1_MAT_s100
                                 + 0.874465 w_k1_MSFT_s100
                                 + 1.06696 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.958138 w_k1_ADM_s100
                                 + 0.807192 w_k1_AXP_s100
                                 + 0.760086 w_k1_DIS_s100
                                 + 0.905307 w_k1_FDX_s100
                                 + 0.843181 w_k1_IBM_s100
                                 + 0.931395 w_k1_INTC_s100
                                 + 0.779398 w_k1_KO_s100
                                 + 0.958492 w_k1_MAT_s100
                                 + 0.873147 w_k1_MSFT_s100
                                 + 0.884566 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.933763 w_k1_ADM_s100
                                 + 1.06211 w_k1_AXP_s100 + 1.3063 w_k1_DIS_s100
                                 + 1.13213 w_k1_FDX_s100
                                 + 1.28901 w_k1_IBM_s100
                                 + 0.998508 w_k1_INTC_s100
                                 + 1.19369 w_k1_KO_s100 + 1.79077 w_k1_MAT_s100
                                 + 1.10164 w_k1_MSFT_s100
                                 + 1.15522 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.86032 w_k1_ADM_s100
                                 + 0.955583 w_k1_AXP_s100
                                 + 0.981219 w_k1_DIS_s100
                                 + 1.04604 w_k1_FDX_s100
                                 + 0.722718 w_k1_IBM_s100
                                 + 0.936777 w_k1_INTC_s100
                                 + 0.820543 w_k1_KO_s100
                                 + 1.03036 w_k1_MAT_s100
                                 + 0.956231 w_k1_MSFT_s100
                                 + 0.98185 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.726858 w_k1_ADM_s100
                                 + 0.646247 w_k1_AXP_s100
                                 + 0.683508 w_k1_DIS_s100
                                 + 0.778246 w_k1_FDX_s100
                                 + 0.622322 w_k1_IBM_s100
                                 + 0.698819 w_k1_INTC_s100
                                 + 0.673792 w_k1_KO_s100
                                 + 0.654916 w_k1_MAT_s100
                                 + 0.717247 w_k1_MSFT_s100
                                 + 0.827583 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.12143 w_k1_ADM_s100
                                 + 1.03314 w_k1_AXP_s100 + 1.1853 w_k1_DIS_s100
                                 + 1.13253 w_k1_FDX_s100
                                 + 0.906743 w_k1_IBM_s100
                                 + 0.836973 w_k1_INTC_s100
                                 + 1.03904 w_k1_KO_s100 + 1.30783 w_k1_MAT_s100
                                 + 1.06336 w_k1_MSFT_s100
                                 + 1.13139 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.71534 w_k1_ADM_s100
                                 + 0.674837 w_k1_AXP_s100
                                 + 0.650859 w_k1_DIS_s100
                                 + 0.698224 w_k1_FDX_s100
                                 + 0.997417 w_k1_IBM_s100
                                 + 0.793026 w_k1_INTC_s100
                                 + 0.820602 w_k1_KO_s100
                                 + 1.08797 w_k1_MAT_s100
                                 + 0.752565 w_k1_MSFT_s100
                                 + 0.866818 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.05699 w_k1_ADM_s101
                                 + 1.08071 w_k1_AXP_s101
                                 + 1.00708 w_k1_DIS_s101
                                 + 1.08248 w_k1_FDX_s101
                                 + 0.771402 w_k1_IBM_s101
                                 + 1.04343 w_k1_INTC_s101
                                 + 0.997854 w_k1_KO_s101
                                 + 0.636742 w_k1_MAT_s101
                                 + 1.139 w_k1_MSFT_s101 + 1.02699 w_k1_PG_s101
                                 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.12538 w_k1_ADM_s101
                                 + 1.47951 w_k1_AXP_s101
                                 + 0.948772 w_k1_DIS_s101
                                 + 0.935443 w_k1_FDX_s101
                                 + 1.13569 w_k1_IBM_s101
                                 + 0.965057 w_k1_INTC_s101
                                 + 1.11229 w_k1_KO_s101 + 1.33122 w_k1_MAT_s101
                                 + 0.92288 w_k1_MSFT_s101
                                 + 1.13808 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.12096 w_k1_ADM_s101
                                 + 1.07931 w_k1_AXP_s101
                                 + 0.867802 w_k1_DIS_s101
                                 + 0.882424 w_k1_FDX_s101
                                 + 1.4612 w_k1_IBM_s101 + 0.9793 w_k1_INTC_s101
                                 + 1.09308 w_k1_KO_s101 + 1.32446 w_k1_MAT_s101
                                 + 0.821239 w_k1_MSFT_s101
                                 + 1.03807 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.710412 w_k1_ADM_s101
                                 + 0.73149 w_k1_AXP_s101
                                 + 0.764531 w_k1_DIS_s101
                                 + 0.732276 w_k1_FDX_s101
                                 + 0.952936 w_k1_IBM_s101
                                 + 0.819146 w_k1_INTC_s101
                                 + 1.01818 w_k1_KO_s101
                                 + 0.841765 w_k1_MAT_s101
                                 + 0.884777 w_k1_MSFT_s101
                                 + 0.900523 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.35472 w_k1_ADM_s101
                                 + 1.41653 w_k1_AXP_s101
                                 + 1.60128 w_k1_DIS_s101
                                 + 1.37338 w_k1_FDX_s101
                                 + 0.984969 w_k1_IBM_s101
                                 + 1.49855 w_k1_INTC_s101
                                 + 1.40462 w_k1_KO_s101
                                 + 0.893782 w_k1_MAT_s101
                                 + 1.19149 w_k1_MSFT_s101
                                 + 1.25108 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.873133 w_k1_ADM_s101
                                 + 1.34477 w_k1_AXP_s101
                                 + 1.48348 w_k1_DIS_s101
                                 + 1.28828 w_k1_FDX_s101 + 1.2831 w_k1_IBM_s101
                                 + 1.1731 w_k1_INTC_s101 + 1.38233 w_k1_KO_s101
                                 + 1.582 w_k1_MAT_s101 + 1.24912 w_k1_MSFT_s101
                                 + 1.18947 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.1217 w_k1_ADM_s101
                                 + 1.29096 w_k1_AXP_s101
                                 + 1.50774 w_k1_DIS_s101
                                 + 1.06853 w_k1_FDX_s101 + 1.7964 w_k1_IBM_s101
                                 + 1.04485 w_k1_INTC_s101
                                 + 1.73263 w_k1_KO_s101 + 2.06824 w_k1_MAT_s101
                                 + 0.931214 w_k1_MSFT_s101
                                 + 1.21457 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.935569 w_k1_ADM_s101
                                 + 0.992165 w_k1_AXP_s101
                                 + 1.28878 w_k1_DIS_s101
                                 + 1.00041 w_k1_FDX_s101 + 1.0741 w_k1_IBM_s101
                                 + 1.09509 w_k1_INTC_s101
                                 + 1.17357 w_k1_KO_s101
                                 + 0.975459 w_k1_MAT_s101
                                 + 1.06652 w_k1_MSFT_s101
                                 + 1.07204 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.734776 w_k1_ADM_s102
                                 + 0.703836 w_k1_AXP_s102
                                 + 0.71752 w_k1_DIS_s102
                                 + 0.749565 w_k1_FDX_s102
                                 + 0.993967 w_k1_IBM_s102
                                 + 0.799223 w_k1_INTC_s102
                                 + 0.985716 w_k1_KO_s102
                                 + 0.627344 w_k1_MAT_s102
                                 + 0.866648 w_k1_MSFT_s102
                                 + 0.909255 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.19466 w_k1_ADM_s102
                                 + 0.759298 w_k1_AXP_s102
                                 + 0.989138 w_k1_DIS_s102
                                 + 1.01156 w_k1_FDX_s102
                                 + 0.654417 w_k1_IBM_s102
                                 + 0.947777 w_k1_INTC_s102
                                 + 0.864882 w_k1_KO_s102
                                 + 0.683825 w_k1_MAT_s102
                                 + 1.09457 w_k1_MSFT_s102
                                 + 1.00179 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.08279 w_k1_ADM_s102
                                 + 0.969566 w_k1_AXP_s102
                                 + 1.33271 w_k1_DIS_s102
                                 + 1.19015 w_k1_FDX_s102
                                 + 0.958587 w_k1_IBM_s102
                                 + 1.19255 w_k1_INTC_s102
                                 + 1.35198 w_k1_KO_s102
                                 + 0.862824 w_k1_MAT_s102
                                 + 1.45175 w_k1_MSFT_s102 + 1.2985 w_k1_PG_s102
                                 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.01127 w_k1_ADM_s102
                                 + 1.07224 w_k1_AXP_s102
                                 + 1.01381 w_k1_DIS_s102
                                 + 0.997715 w_k1_FDX_s102
                                 + 1.51158 w_k1_IBM_s102
                                 + 1.24669 w_k1_INTC_s102
                                 + 1.38102 w_k1_KO_s102 + 1.53609 w_k1_MAT_s102
                                 + 1.26166 w_k1_MSFT_s102
                                 + 1.12195 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.23861 w_k1_ADM_s102
                                 + 1.32728 w_k1_AXP_s102
                                 + 1.23885 w_k1_DIS_s102
                                 + 1.23566 w_k1_FDX_s102
                                 + 1.07604 w_k1_IBM_s102
                                 + 1.12716 w_k1_INTC_s102
                                 + 1.28737 w_k1_KO_s102 + 1.02275 w_k1_MAT_s102
                                 + 1.17299 w_k1_MSFT_s102
                                 + 1.22748 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.880349 w_k1_ADM_s102
                                 + 1.12582 w_k1_AXP_s102
                                 + 0.987011 w_k1_DIS_s102
                                 + 0.971026 w_k1_FDX_s102
                                 + 1.11907 w_k1_IBM_s102
                                 + 0.911644 w_k1_INTC_s102
                                 + 1.03281 w_k1_KO_s102 + 1.02193 w_k1_MAT_s102
                                 + 0.954016 w_k1_MSFT_s102
                                 + 1.07407 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.81504 w_k1_ADM_s102
                                 + 0.62585 w_k1_AXP_s102
                                 + 0.674643 w_k1_DIS_s102
                                 + 0.859472 w_k1_FDX_s102
                                 + 0.761998 w_k1_IBM_s102
                                 + 0.961641 w_k1_INTC_s102
                                 + 0.925512 w_k1_KO_s102
                                 + 0.80105 w_k1_MAT_s102
                                 + 1.03691 w_k1_MSFT_s102
                                 + 0.887241 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.912737 w_k1_ADM_s102
                                 + 0.758263 w_k1_AXP_s102
                                 + 0.960073 w_k1_DIS_s102
                                 + 0.894075 w_k1_FDX_s102
                                 + 0.892323 w_k1_IBM_s102
                                 + 1.16231 w_k1_INTC_s102
                                 + 1.18829 w_k1_KO_s102
                                 + 0.845597 w_k1_MAT_s102
                                 + 1.14685 w_k1_MSFT_s102
                                 + 1.12899 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.767112 w_k1_ADM_s103
                                 + 0.768606 w_k1_AXP_s103
                                 + 0.851309 w_k1_DIS_s103
                                 + 0.798451 w_k1_FDX_s103
                                 + 0.853353 w_k1_IBM_s103
                                 + 0.737164 w_k1_INTC_s103
                                 + 0.736208 w_k1_KO_s103
                                 + 0.729583 w_k1_MAT_s103
                                 + 0.815221 w_k1_MSFT_s103
                                 + 0.86651 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.3211 w_k1_ADM_s103
                                 + 1.46825 w_k1_AXP_s103 + 1.6328 w_k1_DIS_s103
                                 + 1.32932 w_k1_FDX_s103 + 1.0659 w_k1_IBM_s103
                                 + 1.16562 w_k1_INTC_s103 + 1.1222 w_k1_KO_s103
                                 + 1.40094 w_k1_MAT_s103
                                 + 1.33656 w_k1_MSFT_s103
                                 + 1.12214 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.18011 w_k1_ADM_s103
                                 + 1.4005 w_k1_AXP_s103 + 1.66055 w_k1_DIS_s103
                                 + 1.29393 w_k1_FDX_s103 + 1.5876 w_k1_IBM_s103
                                 + 1.22514 w_k1_INTC_s103
                                 + 1.23571 w_k1_KO_s103 + 1.36409 w_k1_MAT_s103
                                 + 1.40806 w_k1_MSFT_s103
                                 + 1.13987 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.11165 w_k1_ADM_s103
                                 + 1.06432 w_k1_AXP_s103
                                 + 0.92612 w_k1_DIS_s103
                                 + 0.975839 w_k1_FDX_s103
                                 + 0.863163 w_k1_IBM_s103
                                 + 1.12443 w_k1_INTC_s103
                                 + 0.81916 w_k1_KO_s103
                                 + 0.856371 w_k1_MAT_s103
                                 + 1.04878 w_k1_MSFT_s103
                                 + 0.97532 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.05309 w_k1_ADM_s103
                                 + 0.972962 w_k1_AXP_s103
                                 + 1.04067 w_k1_DIS_s103
                                 + 0.92689 w_k1_FDX_s103
                                 + 1.30055 w_k1_IBM_s103
                                 + 1.0468 w_k1_INTC_s103 + 1.12642 w_k1_KO_s103
                                 + 1.36987 w_k1_MAT_s103
                                 + 1.07687 w_k1_MSFT_s103
                                 + 0.994977 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.798557 w_k1_ADM_s103
                                 + 0.805293 w_k1_AXP_s103
                                 + 0.78855 w_k1_DIS_s103
                                 + 0.84623 w_k1_FDX_s103
                                 + 0.669087 w_k1_IBM_s103
                                 + 0.732862 w_k1_INTC_s103
                                 + 0.639138 w_k1_KO_s103
                                 + 0.690321 w_k1_MAT_s103
                                 + 0.869972 w_k1_MSFT_s103
                                 + 0.78232 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.95955 w_k1_ADM_s103
                                 + 1.40111 w_k1_AXP_s103
                                 + 1.20617 w_k1_DIS_s103 + 1.1475 w_k1_FDX_s103
                                 + 1.02967 w_k1_IBM_s103
                                 + 1.0516 w_k1_INTC_s103
                                 + 0.925243 w_k1_KO_s103
                                 + 1.36658 w_k1_MAT_s103
                                 + 1.13193 w_k1_MSFT_s103
                                 + 1.00325 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.18683 w_k1_ADM_s103
                                 + 1.10576 w_k1_AXP_s103
                                 + 1.19429 w_k1_DIS_s103
                                 + 1.17057 w_k1_FDX_s103
                                 + 0.77078 w_k1_IBM_s103
                                 + 1.17554 w_k1_INTC_s103
                                 + 0.831967 w_k1_KO_s103
                                 + 0.452199 w_k1_MAT_s103
                                 + 1.06351 w_k1_MSFT_s103
                                 + 0.914439 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.12001 w_k1_ADM_s104
                                 + 1.19012 w_k1_AXP_s104
                                 + 0.843431 w_k1_DIS_s104
                                 + 0.969987 w_k1_FDX_s104
                                 + 0.951348 w_k1_IBM_s104
                                 + 1.0876 w_k1_INTC_s104 + 1.08167 w_k1_KO_s104
                                 + 1.01844 w_k1_MAT_s104
                                 + 1.02841 w_k1_MSFT_s104
                                 + 0.916777 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.44054 w_k1_ADM_s104
                                 + 1.2569 w_k1_AXP_s104 + 1.53888 w_k1_DIS_s104
                                 + 1.32182 w_k1_FDX_s104
                                 + 1.12717 w_k1_IBM_s104
                                 + 0.997483 w_k1_INTC_s104
                                 + 1.48061 w_k1_KO_s104 + 1.13265 w_k1_MAT_s104
                                 + 1.17285 w_k1_MSFT_s104
                                 + 1.08809 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.33241 w_k1_ADM_s104
                                 + 1.53462 w_k1_AXP_s104
                                 + 1.09476 w_k1_DIS_s104
                                 + 1.10546 w_k1_FDX_s104
                                 + 1.62409 w_k1_IBM_s104
                                 + 1.05426 w_k1_INTC_s104
                                 + 1.31453 w_k1_KO_s104 + 1.86985 w_k1_MAT_s104
                                 + 1.13454 w_k1_MSFT_s104
                                 + 1.16512 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.842189 w_k1_ADM_s104
                                 + 0.767555 w_k1_AXP_s104
                                 + 0.792201 w_k1_DIS_s104
                                 + 0.819337 w_k1_FDX_s104
                                 + 1.17899 w_k1_IBM_s104
                                 + 0.854404 w_k1_INTC_s104
                                 + 0.939131 w_k1_KO_s104
                                 + 1.13229 w_k1_MAT_s104
                                 + 0.957987 w_k1_MSFT_s104
                                 + 0.839255 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.890584 w_k1_ADM_s104
                                 + 0.929848 w_k1_AXP_s104
                                 + 1.01208 w_k1_DIS_s104
                                 + 0.99807 w_k1_FDX_s104
                                 + 0.774939 w_k1_IBM_s104
                                 + 1.00739 w_k1_INTC_s104
                                 + 0.888409 w_k1_KO_s104
                                 + 0.560185 w_k1_MAT_s104
                                 + 0.992619 w_k1_MSFT_s104
                                 + 0.821298 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.783598 w_k1_ADM_s104
                                 + 0.669765 w_k1_AXP_s104
                                 + 0.6948 w_k1_DIS_s104
                                 + 0.821494 w_k1_FDX_s104
                                 + 0.691742 w_k1_IBM_s104
                                 + 0.707433 w_k1_INTC_s104
                                 + 0.785866 w_k1_KO_s104
                                 + 0.583697 w_k1_MAT_s104
                                 + 0.960309 w_k1_MSFT_s104
                                 + 0.822435 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.10366 w_k1_ADM_s104
                                 + 1.25538 w_k1_AXP_s104 + 1.0862 w_k1_DIS_s104
                                 + 0.998103 w_k1_FDX_s104
                                 + 1.31755 w_k1_IBM_s104
                                 + 0.970928 w_k1_INTC_s104
                                 + 1.18396 w_k1_KO_s104 + 1.73273 w_k1_MAT_s104
                                 + 0.981279 w_k1_MSFT_s104
                                 + 1.0474 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.977635 w_k1_ADM_s104
                                 + 1.00703 w_k1_AXP_s104
                                 + 1.10028 w_k1_DIS_s104
                                 + 1.04371 w_k1_FDX_s104
                                 + 1.17155 w_k1_IBM_s104
                                 + 1.00083 w_k1_INTC_s104 + 1.2634 w_k1_KO_s104
                                 + 1.06609 w_k1_MAT_s104
                                 + 1.13268 w_k1_MSFT_s104
                                 + 0.964541 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.23156 w_k1_ADM_s105
                                 + 1.24502 w_k1_AXP_s105 + 1.8457 w_k1_DIS_s105
                                 + 1.30503 w_k1_FDX_s105
                                 + 1.03743 w_k1_IBM_s105
                                 + 1.04583 w_k1_INTC_s105
                                 + 1.23522 w_k1_KO_s105
                                 + 0.896358 w_k1_MAT_s105
                                 + 1.36431 w_k1_MSFT_s105 + 1.0674 w_k1_PG_s105
                                 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.908153 w_k1_ADM_s105
                                 + 1.03516 w_k1_AXP_s105
                                 + 1.03824 w_k1_DIS_s105
                                 + 0.913534 w_k1_FDX_s105
                                 + 1.18859 w_k1_IBM_s105
                                 + 0.758293 w_k1_INTC_s105
                                 + 1.04678 w_k1_KO_s105 + 1.43442 w_k1_MAT_s105
                                 + 1.04402 w_k1_MSFT_s105
                                 + 1.00549 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.25214 w_k1_ADM_s105
                                 + 1.54003 w_k1_AXP_s105
                                 + 1.40695 w_k1_DIS_s105
                                 + 1.32656 w_k1_FDX_s105
                                 + 1.26787 w_k1_IBM_s105
                                 + 1.26512 w_k1_INTC_s105 + 1.2179 w_k1_KO_s105
                                 + 1.34436 w_k1_MAT_s105
                                 + 1.34431 w_k1_MSFT_s105
                                 + 1.10484 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.01274 w_k1_ADM_s105
                                 + 0.931479 w_k1_AXP_s105
                                 + 0.802249 w_k1_DIS_s105
                                 + 0.911592 w_k1_FDX_s105
                                 + 0.657691 w_k1_IBM_s105
                                 + 1.04063 w_k1_INTC_s105
                                 + 0.75073 w_k1_KO_s105
                                 + 0.592145 w_k1_MAT_s105
                                 + 1.02567 w_k1_MSFT_s105
                                 + 0.808262 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.03204 w_k1_ADM_s105
                                 + 1.00642 w_k1_AXP_s105
                                 + 1.38747 w_k1_DIS_s105
                                 + 1.15343 w_k1_FDX_s105
                                 + 1.10895 w_k1_IBM_s105
                                 + 1.09096 w_k1_INTC_s105
                                 + 1.09164 w_k1_KO_s105
                                 + 0.726106 w_k1_MAT_s105
                                 + 1.24848 w_k1_MSFT_s105
                                 + 1.01231 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.13764 w_k1_ADM_s105
                                 + 1.2596 w_k1_AXP_s105 + 1.38878 w_k1_DIS_s105
                                 + 1.06296 w_k1_FDX_s105
                                 + 1.63619 w_k1_IBM_s105
                                 + 1.23223 w_k1_INTC_s105
                                 + 1.40938 w_k1_KO_s105 + 1.49291 w_k1_MAT_s105
                                 + 1.27108 w_k1_MSFT_s105
                                 + 1.17749 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.599751 w_k1_ADM_s105
                                 + 0.511405 w_k1_AXP_s105
                                 + 0.717653 w_k1_DIS_s105
                                 + 0.668052 w_k1_FDX_s105
                                 + 1.08991 w_k1_IBM_s105
                                 + 0.819583 w_k1_INTC_s105
                                 + 0.846193 w_k1_KO_s105
                                 + 0.78002 w_k1_MAT_s105
                                 + 0.840215 w_k1_MSFT_s105
                                 + 0.940158 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.742427 w_k1_ADM_s105
                                 + 0.806299 w_k1_AXP_s105
                                 + 0.936582 w_k1_DIS_s105
                                 + 0.82164 w_k1_FDX_s105
                                 + 0.902839 w_k1_IBM_s105
                                 + 0.959516 w_k1_INTC_s105
                                 + 0.91224 w_k1_KO_s105
                                 + 0.795016 w_k1_MAT_s105
                                 + 1.08015 w_k1_MSFT_s105
                                 + 0.973024 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.798476 w_k1_ADM_s106
                                 + 0.825239 w_k1_AXP_s106
                                 + 0.699455 w_k1_DIS_s106
                                 + 0.814459 w_k1_FDX_s106
                                 + 0.753482 w_k1_IBM_s106
                                 + 0.728206 w_k1_INTC_s106
                                 + 0.821579 w_k1_KO_s106
                                 + 0.789787 w_k1_MAT_s106
                                 + 0.767884 w_k1_MSFT_s106
                                 + 0.936558 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.875693 w_k1_ADM_s106
                                 + 1.214 w_k1_AXP_s106 + 1.07095 w_k1_DIS_s106
                                 + 0.997664 w_k1_FDX_s106
                                 + 1.18252 w_k1_IBM_s106
                                 + 1.00317 w_k1_INTC_s106
                                 + 1.14075 w_k1_KO_s106 + 1.20569 w_k1_MAT_s106
                                 + 0.947658 w_k1_MSFT_s106
                                 + 1.12579 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.07519 w_k1_ADM_s106
                                 + 1.02434 w_k1_AXP_s106
                                 + 1.00413 w_k1_DIS_s106
                                 + 1.04978 w_k1_FDX_s106
                                 + 0.816455 w_k1_IBM_s106
                                 + 0.891399 w_k1_INTC_s106
                                 + 0.935651 w_k1_KO_s106
                                 + 0.848598 w_k1_MAT_s106
                                 + 0.937707 w_k1_MSFT_s106
                                 + 1.09372 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.90839 w_k1_ADM_s106
                                 + 0.970069 w_k1_AXP_s106
                                 + 0.987476 w_k1_DIS_s106
                                 + 1.12622 w_k1_FDX_s106
                                 + 0.580764 w_k1_IBM_s106
                                 + 0.828074 w_k1_INTC_s106
                                 + 0.721616 w_k1_KO_s106
                                 + 0.607816 w_k1_MAT_s106
                                 + 0.906872 w_k1_MSFT_s106
                                 + 0.982347 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.900708 w_k1_ADM_s106
                                 + 1.06256 w_k1_AXP_s106
                                 + 1.46489 w_k1_DIS_s106
                                 + 1.10195 w_k1_FDX_s106
                                 + 1.34322 w_k1_IBM_s106
                                 + 1.1153 w_k1_INTC_s106 + 1.3436 w_k1_KO_s106
                                 + 1.39743 w_k1_MAT_s106
                                 + 1.06976 w_k1_MSFT_s106
                                 + 1.13245 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.30338 w_k1_ADM_s106
                                 + 1.37335 w_k1_AXP_s106
                                 + 1.73663 w_k1_DIS_s106
                                 + 1.47675 w_k1_FDX_s106
                                 + 1.21921 w_k1_IBM_s106
                                 + 1.06516 w_k1_INTC_s106
                                 + 1.25377 w_k1_KO_s106 + 1.31685 w_k1_MAT_s106
                                 + 1.23391 w_k1_MSFT_s106
                                 + 1.28141 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.903972 w_k1_ADM_s106
                                 + 1.02399 w_k1_AXP_s106
                                 + 0.960565 w_k1_DIS_s106
                                 + 0.921255 w_k1_FDX_s106
                                 + 1.38386 w_k1_IBM_s106
                                 + 1.0324 w_k1_INTC_s106 + 1.24664 w_k1_KO_s106
                                 + 1.6717 w_k1_MAT_s106
                                 + 0.934914 w_k1_MSFT_s106
                                 + 1.02265 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.03144 w_k1_ADM_s106
                                 + 1.499 w_k1_AXP_s106 + 1.14502 w_k1_DIS_s106
                                 + 1.21029 w_k1_FDX_s106
                                 + 1.05729 w_k1_IBM_s106
                                 + 1.14533 w_k1_INTC_s106
                                 + 1.10499 w_k1_KO_s106 + 1.26291 w_k1_MAT_s106
                                 + 1.1062 w_k1_MSFT_s106 + 1.14399 w_k1_PG_s106
                                 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.803819 w_k1_ADM_s107
                                 + 0.882629 w_k1_AXP_s107
                                 + 1.25261 w_k1_DIS_s107
                                 + 1.02577 w_k1_FDX_s107
                                 + 1.02295 w_k1_IBM_s107
                                 + 1.17612 w_k1_INTC_s107
                                 + 1.23708 w_k1_KO_s107 + 1.14419 w_k1_MAT_s107
                                 + 1.228 w_k1_MSFT_s107 + 0.978688 w_k1_PG_s107
                                 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.03681 w_k1_ADM_s107
                                 + 1.14057 w_k1_AXP_s107
                                 + 1.06413 w_k1_DIS_s107
                                 + 1.16749 w_k1_FDX_s107
                                 + 1.03284 w_k1_IBM_s107
                                 + 1.21894 w_k1_INTC_s107
                                 + 1.14072 w_k1_KO_s107 + 1.60287 w_k1_MAT_s107
                                 + 1.17727 w_k1_MSFT_s107
                                 + 1.30976 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.779559 w_k1_ADM_s107
                                 + 0.859505 w_k1_AXP_s107
                                 + 0.872859 w_k1_DIS_s107
                                 + 0.859103 w_k1_FDX_s107
                                 + 0.978634 w_k1_IBM_s107
                                 + 1.07296 w_k1_INTC_s107
                                 + 0.921063 w_k1_KO_s107
                                 + 1.05219 w_k1_MAT_s107
                                 + 1.08761 w_k1_MSFT_s107
                                 + 0.917619 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.02338 w_k1_ADM_s107
                                 + 1.30064 w_k1_AXP_s107
                                 + 1.11064 w_k1_DIS_s107
                                 + 1.00063 w_k1_FDX_s107
                                 + 1.52593 w_k1_IBM_s107
                                 + 1.12039 w_k1_INTC_s107
                                 + 1.18757 w_k1_KO_s107 + 1.84934 w_k1_MAT_s107
                                 + 0.952221 w_k1_MSFT_s107
                                 + 1.16555 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.783616 w_k1_ADM_s107
                                 + 0.682642 w_k1_AXP_s107
                                 + 0.61548 w_k1_DIS_s107
                                 + 0.77714 w_k1_FDX_s107
                                 + 0.852501 w_k1_IBM_s107
                                 + 0.779416 w_k1_INTC_s107
                                 + 0.77824 w_k1_KO_s107
                                 + 0.735237 w_k1_MAT_s107
                                 + 0.79533 w_k1_MSFT_s107
                                 + 0.959522 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.937594 w_k1_ADM_s107
                                 + 0.974581 w_k1_AXP_s107
                                 + 0.887295 w_k1_DIS_s107
                                 + 1.0456 w_k1_FDX_s107
                                 + 0.693416 w_k1_IBM_s107
                                 + 1.05635 w_k1_INTC_s107
                                 + 0.90471 w_k1_KO_s107
                                 + 0.727193 w_k1_MAT_s107
                                 + 1.01282 w_k1_MSFT_s107
                                 + 0.970026 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.07611 w_k1_ADM_s107
                                 + 1.36624 w_k1_AXP_s107
                                 + 1.31987 w_k1_DIS_s107
                                 + 1.42347 w_k1_FDX_s107
                                 + 0.71501 w_k1_IBM_s107
                                 + 1.19297 w_k1_INTC_s107
                                 + 1.08362 w_k1_KO_s107 + 1.10204 w_k1_MAT_s107
                                 + 1.26893 w_k1_MSFT_s107
                                 + 1.16219 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.39994 w_k1_ADM_s107
                                 + 1.78121 w_k1_AXP_s107
                                 + 1.67619 w_k1_DIS_s107
                                 + 1.43763 w_k1_FDX_s107
                                 + 1.30592 w_k1_IBM_s107
                                 + 1.84864 w_k1_INTC_s107
                                 + 1.58464 w_k1_KO_s107 + 1.37562 w_k1_MAT_s107
                                 + 1.48449 w_k1_MSFT_s107 + 1.323 w_k1_PG_s107
                                 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.18583 w_k1_ADM_s110
                                 + 1.17826 w_k1_AXP_s110 + 1.4572 w_k1_DIS_s110
                                 + 1.18483 w_k1_FDX_s110
                                 + 1.05603 w_k1_IBM_s110
                                 + 1.16012 w_k1_INTC_s110
                                 + 1.04633 w_k1_KO_s110 + 1.30937 w_k1_MAT_s110
                                 + 1.18959 w_k1_MSFT_s110
                                 + 1.21643 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.18004 w_k1_ADM_s110
                                 + 1.20031 w_k1_AXP_s110
                                 + 1.07761 w_k1_DIS_s110
                                 + 1.17544 w_k1_FDX_s110
                                 + 0.913398 w_k1_IBM_s110
                                 + 1.14271 w_k1_INTC_s110
                                 + 0.963045 w_k1_KO_s110
                                 + 0.952306 w_k1_MAT_s110
                                 + 1.11645 w_k1_MSFT_s110
                                 + 1.20365 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.70649 w_k1_ADM_s110
                                 + 0.667892 w_k1_AXP_s110
                                 + 0.671253 w_k1_DIS_s110
                                 + 0.72037 w_k1_FDX_s110
                                 + 0.945749 w_k1_IBM_s110
                                 + 0.762751 w_k1_INTC_s110
                                 + 0.880958 w_k1_KO_s110
                                 + 1.13476 w_k1_MAT_s110
                                 + 0.780196 w_k1_MSFT_s110
                                 + 0.853882 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.20903 w_k1_ADM_s110
                                 + 1.57633 w_k1_AXP_s110
                                 + 1.60758 w_k1_DIS_s110
                                 + 1.41488 w_k1_FDX_s110 + 1.2732 w_k1_IBM_s110
                                 + 1.23054 w_k1_INTC_s110 + 1.2797 w_k1_KO_s110
                                 + 1.44278 w_k1_MAT_s110
                                 + 1.20529 w_k1_MSFT_s110
                                 + 1.31719 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.833684 w_k1_ADM_s110
                                 + 0.93577 w_k1_AXP_s110
                                 + 0.812367 w_k1_DIS_s110
                                 + 0.886724 w_k1_FDX_s110
                                 + 0.882407 w_k1_IBM_s110
                                 + 0.849108 w_k1_INTC_s110
                                 + 0.914513 w_k1_KO_s110
                                 + 1.15022 w_k1_MAT_s110
                                 + 0.883166 w_k1_MSFT_s110
                                 + 1.05764 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.909833 w_k1_ADM_s110
                                 + 1.10419 w_k1_AXP_s110
                                 + 1.16029 w_k1_DIS_s110
                                 + 1.11997 w_k1_FDX_s110 + 1.2647 w_k1_IBM_s110
                                 + 1.15863 w_k1_INTC_s110
                                 + 1.14652 w_k1_KO_s110 + 1.41913 w_k1_MAT_s110
                                 + 1.1087 w_k1_MSFT_s110 + 1.18721 w_k1_PG_s110
                                 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.961364 w_k1_ADM_s110
                                 + 1.08124 w_k1_AXP_s110
                                 + 0.824969 w_k1_DIS_s110
                                 + 0.846036 w_k1_FDX_s110
                                 + 1.2613 w_k1_IBM_s110
                                 + 1.03604 w_k1_INTC_s110
                                 + 1.02802 w_k1_KO_s110 + 1.85613 w_k1_MAT_s110
                                 + 0.916286 w_k1_MSFT_s110
                                 + 0.970723 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.12747 w_k1_ADM_s110
                                 + 1.09865 w_k1_AXP_s110
                                 + 0.988375 w_k1_DIS_s110
                                 + 1.02527 w_k1_FDX_s110
                                 + 1.55428 w_k1_IBM_s110
                                 + 0.849926 w_k1_INTC_s110
                                 + 1.19903 w_k1_KO_s110 + 2.18915 w_k1_MAT_s110
                                 + 0.921366 w_k1_MSFT_s110
                                 + 1.25653 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.900441 w_k1_ADM_s111
                                 + 0.866883 w_k1_AXP_s111
                                 + 1.07925 w_k1_DIS_s111
                                 + 1.05919 w_k1_FDX_s111
                                 + 0.94856 w_k1_IBM_s111
                                 + 1.10534 w_k1_INTC_s111
                                 + 0.879759 w_k1_KO_s111
                                 + 0.66428 w_k1_MAT_s111
                                 + 1.08599 w_k1_MSFT_s111
                                 + 0.970288 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.959516 w_k1_ADM_s111
                                 + 1.42472 w_k1_AXP_s111
                                 + 1.09162 w_k1_DIS_s111
                                 + 0.974013 w_k1_FDX_s111
                                 + 1.56888 w_k1_IBM_s111
                                 + 1.00333 w_k1_INTC_s111
                                 + 1.07307 w_k1_KO_s111 + 1.79831 w_k1_MAT_s111
                                 + 0.897391 w_k1_MSFT_s111
                                 + 0.949731 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.10166 w_k1_ADM_s111
                                 + 1.85505 w_k1_AXP_s111
                                 + 1.74946 w_k1_DIS_s111
                                 + 1.45071 w_k1_FDX_s111 + 1.2157 w_k1_IBM_s111
                                 + 1.23697 w_k1_INTC_s111
                                 + 1.08725 w_k1_KO_s111 + 1.70674 w_k1_MAT_s111
                                 + 1.34093 w_k1_MSFT_s111
                                 + 1.13138 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.08126 w_k1_ADM_s111
                                 + 1.24923 w_k1_AXP_s111
                                 + 1.25422 w_k1_DIS_s111
                                 + 1.18778 w_k1_FDX_s111
                                 + 1.09702 w_k1_IBM_s111
                                 + 1.03284 w_k1_INTC_s111
                                 + 1.03596 w_k1_KO_s111 + 1.26714 w_k1_MAT_s111
                                 + 1.10311 w_k1_MSFT_s111
                                 + 1.08249 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.03109 w_k1_ADM_s111
                                 + 1.024 w_k1_AXP_s111 + 1.16046 w_k1_DIS_s111
                                 + 1.18974 w_k1_FDX_s111
                                 + 0.782717 w_k1_IBM_s111
                                 + 0.983846 w_k1_INTC_s111
                                 + 0.924208 w_k1_KO_s111
                                 + 0.867537 w_k1_MAT_s111
                                 + 1.14373 w_k1_MSFT_s111
                                 + 1.01092 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.01571 w_k1_ADM_s111
                                 + 1.08453 w_k1_AXP_s111 + 1.3684 w_k1_DIS_s111
                                 + 1.12225 w_k1_FDX_s111
                                 + 1.35545 w_k1_IBM_s111
                                 + 0.913409 w_k1_INTC_s111
                                 + 1.08076 w_k1_KO_s111 + 1.60377 w_k1_MAT_s111
                                 + 1.04771 w_k1_MSFT_s111
                                 + 1.04139 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.860279 w_k1_ADM_s111
                                 + 0.93512 w_k1_AXP_s111
                                 + 0.814795 w_k1_DIS_s111
                                 + 0.966561 w_k1_FDX_s111
                                 + 0.678554 w_k1_IBM_s111
                                 + 0.772172 w_k1_INTC_s111
                                 + 0.695584 w_k1_KO_s111
                                 + 0.919931 w_k1_MAT_s111
                                 + 0.840901 w_k1_MSFT_s111
                                 + 0.872136 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.801605 w_k1_ADM_s111
                                 + 0.873749 w_k1_AXP_s111
                                 + 0.706739 w_k1_DIS_s111
                                 + 0.841667 w_k1_FDX_s111
                                 + 0.875459 w_k1_IBM_s111
                                 + 0.807308 w_k1_INTC_s111
                                 + 0.709938 w_k1_KO_s111
                                 + 0.960483 w_k1_MAT_s111
                                 + 0.7301 w_k1_MSFT_s111
                                 + 0.854502 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.10468 w_k1_ADM_s112
                                 + 0.99448 w_k1_AXP_s112
                                 + 0.980422 w_k1_DIS_s112
                                 + 1.03382 w_k1_FDX_s112
                                 + 0.988094 w_k1_IBM_s112
                                 + 1.0973 w_k1_INTC_s112
                                 + 0.949272 w_k1_KO_s112
                                 + 0.836529 w_k1_MAT_s112
                                 + 1.02342 w_k1_MSFT_s112
                                 + 0.986995 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.17461 w_k1_ADM_s112
                                 + 1.29409 w_k1_AXP_s112
                                 + 1.38944 w_k1_DIS_s112
                                 + 1.14852 w_k1_FDX_s112
                                 + 1.77891 w_k1_IBM_s112
                                 + 0.863103 w_k1_INTC_s112
                                 + 1.19092 w_k1_KO_s112 + 1.87591 w_k1_MAT_s112
                                 + 1.09999 w_k1_MSFT_s112
                                 + 1.17972 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.42187 w_k1_ADM_s112
                                 + 1.25016 w_k1_AXP_s112
                                 + 1.37902 w_k1_DIS_s112
                                 + 1.29309 w_k1_FDX_s112
                                 + 0.84073 w_k1_IBM_s112
                                 + 1.18121 w_k1_INTC_s112
                                 + 0.99174 w_k1_KO_s112
                                 + 0.881644 w_k1_MAT_s112
                                 + 1.34207 w_k1_MSFT_s112
                                 + 1.17554 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.974113 w_k1_ADM_s112
                                 + 0.656353 w_k1_AXP_s112
                                 + 0.645388 w_k1_DIS_s112
                                 + 0.808206 w_k1_FDX_s112
                                 + 0.579928 w_k1_IBM_s112
                                 + 0.847813 w_k1_INTC_s112
                                 + 0.665456 w_k1_KO_s112
                                 + 0.632188 w_k1_MAT_s112
                                 + 0.859863 w_k1_MSFT_s112
                                 + 0.820772 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.21495 w_k1_ADM_s112
                                 + 0.819544 w_k1_AXP_s112
                                 + 1.45054 w_k1_DIS_s112
                                 + 1.09255 w_k1_FDX_s112
                                 + 1.21863 w_k1_IBM_s112
                                 + 0.852384 w_k1_INTC_s112 + 1.159 w_k1_KO_s112
                                 + 1.02811 w_k1_MAT_s112
                                 + 1.05945 w_k1_MSFT_s112
                                 + 1.14806 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.02801 w_k1_ADM_s112
                                 + 1.53527 w_k1_AXP_s112
                                 + 1.19657 w_k1_DIS_s112
                                 + 1.04451 w_k1_FDX_s112
                                 + 1.08968 w_k1_IBM_s112
                                 + 1.02286 w_k1_INTC_s112
                                 + 0.99519 w_k1_KO_s112 + 1.12584 w_k1_MAT_s112
                                 + 0.981658 w_k1_MSFT_s112
                                 + 0.961717 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.19661 w_k1_ADM_s112
                                 + 1.5388 w_k1_AXP_s112 + 1.43213 w_k1_DIS_s112
                                 + 1.27054 w_k1_FDX_s112
                                 + 1.45636 w_k1_IBM_s112
                                 + 1.21079 w_k1_INTC_s112
                                 + 1.28527 w_k1_KO_s112 + 1.6052 w_k1_MAT_s112
                                 + 1.15125 w_k1_MSFT_s112
                                 + 1.26344 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.881445 w_k1_ADM_s112
                                 + 0.67447 w_k1_AXP_s112
                                 + 0.686433 w_k1_DIS_s112
                                 + 0.774385 w_k1_FDX_s112
                                 + 0.985833 w_k1_IBM_s112
                                 + 0.732714 w_k1_INTC_s112
                                 + 0.791669 w_k1_KO_s112
                                 + 0.994351 w_k1_MAT_s112
                                 + 0.809925 w_k1_MSFT_s112
                                 + 0.856558 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.963176 w_k1_ADM_s113
                                 + 1.12574 w_k1_AXP_s113
                                 + 1.08491 w_k1_DIS_s113
                                 + 1.09463 w_k1_FDX_s113
                                 + 0.956736 w_k1_IBM_s113
                                 + 0.985365 w_k1_INTC_s113
                                 + 1.15459 w_k1_KO_s113 + 1.58046 w_k1_MAT_s113
                                 + 1.13198 w_k1_MSFT_s113
                                 + 1.06677 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.00791 w_k1_ADM_s113
                                 + 1.26165 w_k1_AXP_s113
                                 + 1.08539 w_k1_DIS_s113
                                 + 1.21805 w_k1_FDX_s113
                                 + 0.865185 w_k1_IBM_s113
                                 + 1.18262 w_k1_INTC_s113
                                 + 0.966736 w_k1_KO_s113
                                 + 1.12564 w_k1_MAT_s113
                                 + 1.07305 w_k1_MSFT_s113
                                 + 0.937612 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.993122 w_k1_ADM_s113
                                 + 0.777104 w_k1_AXP_s113
                                 + 0.780628 w_k1_DIS_s113
                                 + 1.03262 w_k1_FDX_s113
                                 + 0.69491 w_k1_IBM_s113
                                 + 0.832441 w_k1_INTC_s113
                                 + 0.781645 w_k1_KO_s113
                                 + 0.875586 w_k1_MAT_s113
                                 + 0.881573 w_k1_MSFT_s113
                                 + 0.88339 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.887903 w_k1_ADM_s113
                                 + 1.06561 w_k1_AXP_s113
                                 + 0.885647 w_k1_DIS_s113
                                 + 0.967264 w_k1_FDX_s113
                                 + 1.02938 w_k1_IBM_s113
                                 + 0.857492 w_k1_INTC_s113
                                 + 0.956202 w_k1_KO_s113 + 1.2477 w_k1_MAT_s113
                                 + 0.916556 w_k1_MSFT_s113
                                 + 0.993235 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.697379 w_k1_ADM_s113
                                 + 0.82696 w_k1_AXP_s113
                                 + 0.590263 w_k1_DIS_s113
                                 + 0.787608 w_k1_FDX_s113
                                 + 0.786795 w_k1_IBM_s113
                                 + 0.790111 w_k1_INTC_s113
                                 + 0.714483 w_k1_KO_s113
                                 + 0.929229 w_k1_MAT_s113
                                 + 0.850647 w_k1_MSFT_s113
                                 + 0.875334 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.01183 w_k1_ADM_s113
                                 + 0.92014 w_k1_AXP_s113
                                 + 1.25115 w_k1_DIS_s113
                                 + 1.01396 w_k1_FDX_s113
                                 + 1.39641 w_k1_IBM_s113
                                 + 1.00321 w_k1_INTC_s113
                                 + 1.14513 w_k1_KO_s113 + 1.06312 w_k1_MAT_s113
                                 + 1.0536 w_k1_MSFT_s113 + 1.03907 w_k1_PG_s113
                                 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.61423 w_k1_ADM_s113
                                 + 2.04349 w_k1_AXP_s113
                                 + 1.98097 w_k1_DIS_s113
                                 + 1.59464 w_k1_FDX_s113
                                 + 1.40543 w_k1_IBM_s113
                                 + 1.11038 w_k1_INTC_s113
                                 + 1.51296 w_k1_KO_s113 + 1.63076 w_k1_MAT_s113
                                 + 1.28432 w_k1_MSFT_s113
                                 + 1.38526 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.33695 w_k1_ADM_s113
                                 + 1.3039 w_k1_AXP_s113 + 1.37473 w_k1_DIS_s113
                                 + 1.35097 w_k1_FDX_s113
                                 + 1.04482 w_k1_IBM_s113
                                 + 1.12578 w_k1_INTC_s113
                                 + 1.10681 w_k1_KO_s113 + 1.28391 w_k1_MAT_s113
                                 + 1.26458 w_k1_MSFT_s113
                                 + 1.22172 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.961337 w_k1_ADM_s114
                                 + 0.783933 w_k1_AXP_s114
                                 + 0.648761 w_k1_DIS_s114
                                 + 0.742139 w_k1_FDX_s114
                                 + 0.703025 w_k1_IBM_s114
                                 + 0.899834 w_k1_INTC_s114
                                 + 0.768408 w_k1_KO_s114
                                 + 0.670744 w_k1_MAT_s114
                                 + 0.839926 w_k1_MSFT_s114
                                 + 0.836639 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.928436 w_k1_ADM_s114
                                 + 1.19465 w_k1_AXP_s114
                                 + 1.12773 w_k1_DIS_s114
                                 + 0.934212 w_k1_FDX_s114
                                 + 1.57545 w_k1_IBM_s114
                                 + 1.0604 w_k1_INTC_s114 + 1.24793 w_k1_KO_s114
                                 + 1.64896 w_k1_MAT_s114
                                 + 1.13025 w_k1_MSFT_s114 + 1.1669 w_k1_PG_s114
                                 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.914599 w_k1_ADM_s114
                                 + 0.760623 w_k1_AXP_s114
                                 + 0.740048 w_k1_DIS_s114
                                 + 0.823695 w_k1_FDX_s114
                                 + 0.958417 w_k1_IBM_s114
                                 + 0.722544 w_k1_INTC_s114
                                 + 0.85691 w_k1_KO_s114 + 1.19974 w_k1_MAT_s114
                                 + 0.950754 w_k1_MSFT_s114
                                 + 0.980487 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.06841 w_k1_ADM_s114
                                 + 1.3384 w_k1_AXP_s114 + 1.20968 w_k1_DIS_s114
                                 + 1.20333 w_k1_FDX_s114
                                 + 1.15437 w_k1_IBM_s114
                                 + 1.22355 w_k1_INTC_s114
                                 + 1.28046 w_k1_KO_s114 + 1.31839 w_k1_MAT_s114
                                 + 1.21305 w_k1_MSFT_s114
                                 + 1.06725 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.00095 w_k1_ADM_s114
                                 + 1.04927 w_k1_AXP_s114
                                 + 0.993255 w_k1_DIS_s114
                                 + 1.06094 w_k1_FDX_s114
                                 + 0.673225 w_k1_IBM_s114
                                 + 0.904862 w_k1_INTC_s114
                                 + 0.965259 w_k1_KO_s114
                                 + 1.10645 w_k1_MAT_s114
                                 + 1.15243 w_k1_MSFT_s114
                                 + 0.90519 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.45965 w_k1_ADM_s114
                                 + 1.38166 w_k1_AXP_s114
                                 + 1.75562 w_k1_DIS_s114
                                 + 1.26988 w_k1_FDX_s114
                                 + 1.38936 w_k1_IBM_s114
                                 + 1.18888 w_k1_INTC_s114
                                 + 1.61456 w_k1_KO_s114 + 1.43872 w_k1_MAT_s114
                                 + 1.40362 w_k1_MSFT_s114
                                 + 1.39141 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.973665 w_k1_ADM_s114
                                 + 1.10484 w_k1_AXP_s114
                                 + 0.944968 w_k1_DIS_s114
                                 + 0.990306 w_k1_FDX_s114
                                 + 1.04237 w_k1_IBM_s114
                                 + 0.951737 w_k1_INTC_s114
                                 + 1.12159 w_k1_KO_s114 + 1.28531 w_k1_MAT_s114
                                 + 1.2222 w_k1_MSFT_s114 + 1.05773 w_k1_PG_s114
                                 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.29062 w_k1_ADM_s114
                                 + 1.2613 w_k1_AXP_s114 + 1.36541 w_k1_DIS_s114
                                 + 1.17998 w_k1_FDX_s114
                                 + 0.828246 w_k1_IBM_s114
                                 + 1.32324 w_k1_INTC_s114
                                 + 1.11064 w_k1_KO_s114
                                 + 0.587459 w_k1_MAT_s114
                                 + 1.3239 w_k1_MSFT_s114 + 1.10512 w_k1_PG_s114
                                 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.711348 w_k1_ADM_s115
                                 + 0.743695 w_k1_AXP_s115
                                 + 0.746839 w_k1_DIS_s115
                                 + 0.776659 w_k1_FDX_s115
                                 + 0.882327 w_k1_IBM_s115
                                 + 0.871033 w_k1_INTC_s115
                                 + 0.743178 w_k1_KO_s115
                                 + 0.674214 w_k1_MAT_s115
                                 + 0.730497 w_k1_MSFT_s115
                                 + 0.724253 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.13 w_k1_ADM_s115
                                 + 1.22402 w_k1_AXP_s115 + 1.6983 w_k1_DIS_s115
                                 + 1.08501 w_k1_FDX_s115
                                 + 1.70106 w_k1_IBM_s115
                                 + 1.00844 w_k1_INTC_s115
                                 + 1.72633 w_k1_KO_s115 + 1.57695 w_k1_MAT_s115
                                 + 1.13922 w_k1_MSFT_s115
                                 + 1.23748 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.926904 w_k1_ADM_s115
                                 + 0.684162 w_k1_AXP_s115
                                 + 0.819943 w_k1_DIS_s115
                                 + 0.803634 w_k1_FDX_s115
                                 + 0.653323 w_k1_IBM_s115
                                 + 0.688515 w_k1_INTC_s115
                                 + 0.70978 w_k1_KO_s115
                                 + 0.560751 w_k1_MAT_s115
                                 + 0.736883 w_k1_MSFT_s115
                                 + 0.82054 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.34804 w_k1_ADM_s115
                                 + 1.46477 w_k1_AXP_s115
                                 + 1.76727 w_k1_DIS_s115
                                 + 1.36893 w_k1_FDX_s115 + 1.1644 w_k1_IBM_s115
                                 + 1.25155 w_k1_INTC_s115 + 1.4417 w_k1_KO_s115
                                 + 1.41233 w_k1_MAT_s115
                                 + 1.32875 w_k1_MSFT_s115
                                 + 1.31506 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.02183 w_k1_ADM_s115
                                 + 1.30803 w_k1_AXP_s115
                                 + 1.40836 w_k1_DIS_s115
                                 + 1.15915 w_k1_FDX_s115
                                 + 1.02779 w_k1_IBM_s115
                                 + 1.16131 w_k1_INTC_s115
                                 + 1.18571 w_k1_KO_s115 + 1.00257 w_k1_MAT_s115
                                 + 1.08207 w_k1_MSFT_s115
                                 + 1.13668 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.81077 w_k1_ADM_s115
                                 + 1.00373 w_k1_AXP_s115
                                 + 1.05755 w_k1_DIS_s115
                                 + 0.980012 w_k1_FDX_s115
                                 + 1.26557 w_k1_IBM_s115
                                 + 1.11173 w_k1_INTC_s115
                                 + 1.12537 w_k1_KO_s115 + 1.16699 w_k1_MAT_s115
                                 + 0.941002 w_k1_MSFT_s115
                                 + 0.933511 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.10565 w_k1_ADM_s115
                                 + 1.00367 w_k1_AXP_s115
                                 + 1.27181 w_k1_DIS_s115
                                 + 1.16969 w_k1_FDX_s115
                                 + 0.785067 w_k1_IBM_s115
                                 + 0.921918 w_k1_INTC_s115
                                 + 0.986623 w_k1_KO_s115
                                 + 0.780161 w_k1_MAT_s115
                                 + 1.0581 w_k1_MSFT_s115
                                 + 0.980278 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.06633 w_k1_ADM_s115
                                 + 1.06103 w_k1_AXP_s115
                                 + 1.01371 w_k1_DIS_s115
                                 + 1.01331 w_k1_FDX_s115
                                 + 0.981516 w_k1_IBM_s115
                                 + 0.992286 w_k1_INTC_s115
                                 + 0.950651 w_k1_KO_s115
                                 + 0.746812 w_k1_MAT_s115
                                 + 0.89081 w_k1_MSFT_s115
                                 + 0.99982 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.922325 w_k1_ADM_s116
                                 + 0.829765 w_k1_AXP_s116
                                 + 0.783663 w_k1_DIS_s116
                                 + 0.842956 w_k1_FDX_s116
                                 + 0.763314 w_k1_IBM_s116
                                 + 0.833199 w_k1_INTC_s116
                                 + 0.848922 w_k1_KO_s116
                                 + 0.749509 w_k1_MAT_s116
                                 + 0.74002 w_k1_MSFT_s116
                                 + 0.774141 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.34412 w_k1_ADM_s116
                                 + 1.64763 w_k1_AXP_s116
                                 + 2.10605 w_k1_DIS_s116
                                 + 1.68197 w_k1_FDX_s116
                                 + 1.15808 w_k1_IBM_s116
                                 + 1.10497 w_k1_INTC_s116 + 1.4127 w_k1_KO_s116
                                 + 1.59097 w_k1_MAT_s116
                                 + 1.48814 w_k1_MSFT_s116
                                 + 1.17292 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.13352 w_k1_ADM_s116
                                 + 1.13164 w_k1_AXP_s116
                                 + 1.06844 w_k1_DIS_s116
                                 + 1.12913 w_k1_FDX_s116
                                 + 0.740747 w_k1_IBM_s116
                                 + 0.805976 w_k1_INTC_s116
                                 + 0.866637 w_k1_KO_s116
                                 + 1.11129 w_k1_MAT_s116
                                 + 0.936613 w_k1_MSFT_s116
                                 + 0.834752 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.00568 w_k1_ADM_s116
                                 + 1.07216 w_k1_AXP_s116
                                 + 0.967538 w_k1_DIS_s116
                                 + 0.970164 w_k1_FDX_s116
                                 + 1.10725 w_k1_IBM_s116
                                 + 0.939024 w_k1_INTC_s116
                                 + 1.05782 w_k1_KO_s116 + 1.15077 w_k1_MAT_s116
                                 + 0.95729 w_k1_MSFT_s116
                                 + 0.963993 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.63747 w_k1_ADM_s116
                                 + 0.754681 w_k1_AXP_s116
                                 + 0.668511 w_k1_DIS_s116
                                 + 0.718558 w_k1_FDX_s116
                                 + 1.03956 w_k1_IBM_s116
                                 + 0.757417 w_k1_INTC_s116
                                 + 0.903305 w_k1_KO_s116
                                 + 1.20045 w_k1_MAT_s116
                                 + 0.794282 w_k1_MSFT_s116
                                 + 0.819012 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.48343 w_k1_ADM_s116
                                 + 1.17789 w_k1_AXP_s116
                                 + 1.33427 w_k1_DIS_s116 + 1.0833 w_k1_FDX_s116
                                 + 1.50758 w_k1_IBM_s116
                                 + 0.95118 w_k1_INTC_s116
                                 + 1.24292 w_k1_KO_s116 + 1.74923 w_k1_MAT_s116
                                 + 1.10257 w_k1_MSFT_s116
                                 + 1.07986 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.2137 w_k1_ADM_s116
                                 + 1.13376 w_k1_AXP_s116
                                 + 1.38364 w_k1_DIS_s116
                                 + 1.13125 w_k1_FDX_s116
                                 + 0.803956 w_k1_IBM_s116
                                 + 0.987019 w_k1_INTC_s116
                                 + 1.06764 w_k1_KO_s116
                                 + 0.662908 w_k1_MAT_s116
                                 + 1.11859 w_k1_MSFT_s116
                                 + 0.972379 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.36589 w_k1_ADM_s116
                                 + 1.3892 w_k1_AXP_s116 + 1.13308 w_k1_DIS_s116
                                 + 1.17142 w_k1_FDX_s116
                                 + 0.928622 w_k1_IBM_s116
                                 + 0.890038 w_k1_INTC_s116
                                 + 1.13496 w_k1_KO_s116 + 1.14459 w_k1_MAT_s116
                                 + 1.03985 w_k1_MSFT_s116
                                 + 1.05804 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.12756 w_k1_ADM_s117
                                 + 1.30994 w_k1_AXP_s117
                                 + 1.33615 w_k1_DIS_s117
                                 + 1.13176 w_k1_FDX_s117
                                 + 1.59969 w_k1_IBM_s117
                                 + 1.16984 w_k1_INTC_s117
                                 + 1.50297 w_k1_KO_s117 + 1.73357 w_k1_MAT_s117
                                 + 1.32402 w_k1_MSFT_s117
                                 + 1.11315 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.810383 w_k1_ADM_s117
                                 + 0.65157 w_k1_AXP_s117
                                 + 0.774935 w_k1_DIS_s117
                                 + 0.748959 w_k1_FDX_s117
                                 + 0.697693 w_k1_IBM_s117
                                 + 0.786095 w_k1_INTC_s117
                                 + 0.737802 w_k1_KO_s117
                                 + 0.600072 w_k1_MAT_s117
                                 + 0.850277 w_k1_MSFT_s117
                                 + 0.805464 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.1412 w_k1_ADM_s117
                                 + 1.06482 w_k1_AXP_s117
                                 + 0.977297 w_k1_DIS_s117
                                 + 1.09903 w_k1_FDX_s117
                                 + 0.680342 w_k1_IBM_s117
                                 + 0.887079 w_k1_INTC_s117
                                 + 0.745868 w_k1_KO_s117
                                 + 0.713502 w_k1_MAT_s117
                                 + 1.05584 w_k1_MSFT_s117
                                 + 0.934673 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.0706 w_k1_ADM_s117
                                 + 1.24486 w_k1_AXP_s117
                                 + 0.877628 w_k1_DIS_s117
                                 + 1.01712 w_k1_FDX_s117
                                 + 0.852296 w_k1_IBM_s117
                                 + 0.90958 w_k1_INTC_s117
                                 + 0.861915 w_k1_KO_s117
                                 + 1.06782 w_k1_MAT_s117
                                 + 0.947682 w_k1_MSFT_s117
                                 + 0.98406 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.19907 w_k1_ADM_s117
                                 + 1.37235 w_k1_AXP_s117
                                 + 1.44833 w_k1_DIS_s117 + 1.194 w_k1_FDX_s117
                                 + 1.10808 w_k1_IBM_s117
                                 + 1.07367 w_k1_INTC_s117
                                 + 1.04676 w_k1_KO_s117 + 1.0431 w_k1_MAT_s117
                                 + 1.10629 w_k1_MSFT_s117
                                 + 0.998948 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.00106 w_k1_ADM_s117
                                 + 0.988397 w_k1_AXP_s117
                                 + 0.957699 w_k1_DIS_s117
                                 + 0.870239 w_k1_FDX_s117
                                 + 1.24254 w_k1_IBM_s117
                                 + 0.900264 w_k1_INTC_s117
                                 + 0.966976 w_k1_KO_s117
                                 + 0.914899 w_k1_MAT_s117
                                 + 0.882721 w_k1_MSFT_s117
                                 + 0.943777 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.75946 w_k1_ADM_s117
                                 + 0.66476 w_k1_AXP_s117
                                 + 0.764492 w_k1_DIS_s117
                                 + 0.71578 w_k1_FDX_s117
                                 + 1.02956 w_k1_IBM_s117
                                 + 0.740185 w_k1_INTC_s117
                                 + 0.942063 w_k1_KO_s117
                                 + 0.899259 w_k1_MAT_s117
                                 + 0.797156 w_k1_MSFT_s117
                                 + 0.839669 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.891923 w_k1_ADM_s117
                                 + 0.912857 w_k1_AXP_s117
                                 + 0.927578 w_k1_DIS_s117
                                 + 0.901484 w_k1_FDX_s117
                                 + 0.831148 w_k1_IBM_s117
                                 + 1.0054 w_k1_INTC_s117
                                 + 0.909778 w_k1_KO_s117
                                 + 0.773013 w_k1_MAT_s117
                                 + 0.988661 w_k1_MSFT_s117
                                 + 0.86846 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0206877894682483 w_k1_ADM_s000
                                 + 0.0415706344383378 w_k1_ADM_s001
                                 + 0.0628875803164334 w_k1_ADM_s002
                                 + 0.0277721015363896 w_k1_ADM_s003
                                 + 0.0625768979373138 w_k1_ADM_s004
                                 + 0.0245194457679299 w_k1_ADM_s005
                                 + 0.0263568187168595 w_k1_ADM_s006
                                 + 0.0342932358874119 w_k1_ADM_s007
                                 + 0.04158615379922 w_k1_ADM_s010
                                 + 0.0220006889844364 w_k1_ADM_s011
                                 + 0.0369067648823374 w_k1_ADM_s012
                                 + 0.0138514310787933 w_k1_ADM_s013
                                 + 0.0198734616840418 w_k1_ADM_s014
                                 + 0.0569145396021054 w_k1_ADM_s015
                                 + 0.0171775309209981 w_k1_ADM_s016
                                 + 0.0199120987302002 w_k1_ADM_s017
                                 + 0.051143236077828 w_k1_ADM_s100
                                 + 0.0302372635305791 w_k1_ADM_s101
                                 + 0.0235083685523732 w_k1_ADM_s102
                                 + 0.0334373703766817 w_k1_ADM_s103
                                 + 0.0338526321475366 w_k1_ADM_s104
                                 + 0.0186454924990882 w_k1_ADM_s105
                                 + 0.0376749349405557 w_k1_ADM_s106
                                 + 0.0312883873236099 w_k1_ADM_s107
                                 + 0.023341573370312 w_k1_ADM_s110
                                 + 0.0314921738541736 w_k1_ADM_s111
                                 + 0.0604874222021682 w_k1_ADM_s112
                                 + 0.016626743374951 w_k1_ADM_s113
                                 + 0.0138274476619203 w_k1_ADM_s114
                                 + 0.0132168554513236 w_k1_ADM_s115
                                 + 0.0390992231201265 w_k1_ADM_s116
                                 + 0.0259852676601659 w_k1_ADM_s117
                                 + 0.024928588387837 w_k1_AXP_s000
                                 + 0.0414057788829351 w_k1_AXP_s001
                                 + 0.0569512067562686 w_k1_AXP_s002
                                 + 0.0266843828839156 w_k1_AXP_s003
                                 + 0.064998027634523 w_k1_AXP_s004
                                 + 0.026290792792765 w_k1_AXP_s005
                                 + 0.0254905119260014 w_k1_AXP_s006
                                 + 0.0346472995419933 w_k1_AXP_s007
                                 + 0.051691346012232 w_k1_AXP_s010
                                 + 0.0257919062226882 w_k1_AXP_s011
                                 + 0.037759545182191 w_k1_AXP_s012
                                 + 0.0149972377394911 w_k1_AXP_s013
                                 + 0.0171196053601473 w_k1_AXP_s014
                                 + 0.0519849431793037 w_k1_AXP_s015
                                 + 0.0169873578889258 w_k1_AXP_s016
                                 + 0.0174131004484459 w_k1_AXP_s017
                                 + 0.0504517506903415 w_k1_AXP_s100
                                 + 0.0334434718589984 w_k1_AXP_s101
                                 + 0.0231462612224515 w_k1_AXP_s102
                                 + 0.0349148772312534 w_k1_AXP_s103
                                 + 0.0335210543777855 w_k1_AXP_s104
                                 + 0.0194520837181066 w_k1_AXP_s105
                                 + 0.0431374198228072 w_k1_AXP_s106
                                 + 0.0355916645642166 w_k1_AXP_s107
                                 + 0.0255946316080575 w_k1_AXP_s110
                                 + 0.0368433013918745 w_k1_AXP_s111
                                 + 0.0553527777527619 w_k1_AXP_s112
                                 + 0.0184844095170258 w_k1_AXP_s113
                                 + 0.0139894922968595 w_k1_AXP_s114
                                 + 0.0136076968208332 w_k1_AXP_s115
                                 + 0.0382730545687719 w_k1_AXP_s116
                                 + 0.0273761963014977 w_k1_AXP_s117
                                 + 0.0238476947913783 w_k1_DIS_s000
                                 + 0.0397033129616444 w_k1_DIS_s001
                                 + 0.0606911255658269 w_k1_DIS_s002
                                 + 0.0289493129876493 w_k1_DIS_s003
                                 + 0.065475342867143 w_k1_DIS_s004
                                 + 0.028931815450321 w_k1_DIS_s005
                                 + 0.0271081963988148 w_k1_DIS_s006
                                 + 0.034660666936486 w_k1_DIS_s007
                                 + 0.0496084256285883 w_k1_DIS_s010
                                 + 0.0259998982851265 w_k1_DIS_s011
                                 + 0.0357355987028764 w_k1_DIS_s012
                                 + 0.0147939492115247 w_k1_DIS_s013
                                 + 0.0165274969450634 w_k1_DIS_s014
                                 + 0.0522713636649506 w_k1_DIS_s015
                                 + 0.020338075432251 w_k1_DIS_s016
                                 + 0.0170605163117921 w_k1_DIS_s017
                                 + 0.0564363289887838 w_k1_DIS_s100
                                 + 0.0338040238839437 w_k1_DIS_s101
                                 + 0.0240726970127069 w_k1_DIS_s102
                                 + 0.0358487402115377 w_k1_DIS_s103
                                 + 0.0333537691018521 w_k1_DIS_s104
                                 + 0.0218625837450757 w_k1_DIS_s105
                                 + 0.0427995784727138 w_k1_DIS_s106
                                 + 0.0338961635823102 w_k1_DIS_s107
                                 + 0.024331363464598 w_k1_DIS_s110
                                 + 0.0356393184792843 w_k1_DIS_s111
                                 + 0.0567284170869881 w_k1_DIS_s112
                                 + 0.0173308512256336 w_k1_DIS_s113
                                 + 0.0137098055351724 w_k1_DIS_s114
                                 + 0.0154929798768287 w_k1_DIS_s115
                                 + 0.0385292792024201 w_k1_DIS_s116
                                 + 0.0275876630440266 w_k1_DIS_s117
                                 + 0.0241588491393053 w_k1_FDX_s000
                                 + 0.0423978418513936 w_k1_FDX_s001
                                 + 0.0565544810154944 w_k1_FDX_s002
                                 + 0.0285444379248849 w_k1_FDX_s003
                                 + 0.0653578028778774 w_k1_FDX_s004
                                 + 0.0274153935069632 w_k1_FDX_s005
                                 + 0.0266990588318632 w_k1_FDX_s006
                                 + 0.0333565399901141 w_k1_FDX_s007
                                 + 0.0486377934404479 w_k1_FDX_s010
                                 + 0.026223374507119 w_k1_FDX_s011
                                 + 0.0354561566318173 w_k1_FDX_s012
                                 + 0.0138062679141481 w_k1_FDX_s013
                                 + 0.0161475098905 w_k1_FDX_s014
                                 + 0.0517168114948015 w_k1_FDX_s015
                                 + 0.0181297144795318 w_k1_FDX_s016
                                 + 0.0174301486663685 w_k1_FDX_s017
                                 + 0.0532387752289443 w_k1_FDX_s100
                                 + 0.0309083160251715 w_k1_FDX_s101
                                 + 0.024098643928996 w_k1_FDX_s102
                                 + 0.0333330562677073 w_k1_FDX_s103
                                 + 0.0327980875466856 w_k1_FDX_s104
                                 + 0.0190300070613815 w_k1_FDX_s105
                                 + 0.0414705406082848 w_k1_FDX_s106
                                 + 0.0342329141460416 w_k1_FDX_s107
                                 + 0.0243523200058619 w_k1_FDX_s110
                                 + 0.0352673588893476 w_k1_FDX_s111
                                 + 0.0559395924652977 w_k1_FDX_s112
                                 + 0.0178427922213848 w_k1_FDX_s113
                                 + 0.0131873262716972 w_k1_FDX_s114
                                 + 0.0134294110902117 w_k1_FDX_s115
                                 + 0.0364772168637001 w_k1_FDX_s116
                                 + 0.0250660300902098 w_k1_FDX_s117
                                 + 0.0242430164206269 w_k1_IBM_s000
                                 + 0.0424838116784079 w_k1_IBM_s001
                                 + 0.0610096475771256 w_k1_IBM_s002
                                 + 0.0286529458488561 w_k1_IBM_s003
                                 + 0.0649847145419949 w_k1_IBM_s004
                                 + 0.0260847679405168 w_k1_IBM_s005
                                 + 0.0271498537470592 w_k1_IBM_s006
                                 + 0.0328333615953853 w_k1_IBM_s007
                                 + 0.0502351903259468 w_k1_IBM_s010
                                 + 0.0276855982143929 w_k1_IBM_s011
                                 + 0.0348473259598822 w_k1_IBM_s012
                                 + 0.0134106502638422 w_k1_IBM_s013
                                 + 0.016949839057362 w_k1_IBM_s014
                                 + 0.0520488587658726 w_k1_IBM_s015
                                 + 0.0182670082826987 w_k1_IBM_s016
                                 + 0.017155035294015 w_k1_IBM_s017
                                 + 0.0527190518754547 w_k1_IBM_s100
                                 + 0.0322194102345129 w_k1_IBM_s101
                                 + 0.0250732419680144 w_k1_IBM_s102
                                 + 0.0337015432649235 w_k1_IBM_s103
                                 + 0.0326372690029384 w_k1_IBM_s104
                                 + 0.0199646186004576 w_k1_IBM_s105
                                 + 0.041361663211137 w_k1_IBM_s106
                                 + 0.0331201331990441 w_k1_IBM_s107
                                 + 0.0256650395744198 w_k1_IBM_s110
                                 + 0.0335957569680029 w_k1_IBM_s111
                                 + 0.0585677799957542 w_k1_IBM_s112
                                 + 0.0167852779061262 w_k1_IBM_s113
                                 + 0.0134873815751812 w_k1_IBM_s114
                                 + 0.0140199272759214 w_k1_IBM_s115
                                 + 0.0358741882542442 w_k1_IBM_s116
                                 + 0.0276261818809464 w_k1_IBM_s117
                                 + 0.0229328684724509 w_k1_INTC_s000
                                 + 0.0418775084498078 w_k1_INTC_s001
                                 + 0.0541219515814218 w_k1_INTC_s002
                                 + 0.0256960238481507 w_k1_INTC_s003
                                 + 0.0649713860662653 w_k1_INTC_s004
                                 + 0.0287096701120417 w_k1_INTC_s005
                                 + 0.0258691745339949 w_k1_INTC_s006
                                 + 0.029998482832342 w_k1_INTC_s007
                                 + 0.0534277459977061 w_k1_INTC_s010
                                 + 0.0256847932737966 w_k1_INTC_s011
                                 + 0.0368965490658778 w_k1_INTC_s012
                                 + 0.0137919781681481 w_k1_INTC_s013
                                 + 0.016340348547582 w_k1_INTC_s014
                                 + 0.0457818417814749 w_k1_INTC_s015
                                 + 0.0198321638092833 w_k1_INTC_s016
                                 + 0.0170379341662818 w_k1_INTC_s017
                                 + 0.046416781483414 w_k1_INTC_s100
                                 + 0.03192399464269 w_k1_INTC_s101
                                 + 0.0250825878986573 w_k1_INTC_s102
                                 + 0.033036454311363 w_k1_INTC_s103
                                 + 0.0308503774160176 w_k1_INTC_s104
                                 + 0.0197434993126621 w_k1_INTC_s105
                                 + 0.0380563239436764 w_k1_INTC_s106
                                 + 0.0374447760930374 w_k1_INTC_s107
                                 + 0.0239570631709701 w_k1_INTC_s110
                                 + 0.0314680611151325 w_k1_INTC_s111
                                 + 0.0534521984748697 w_k1_INTC_s112
                                 + 0.0159037119535726 w_k1_INTC_s113
                                 + 0.0128467991978571 w_k1_INTC_s114
                                 + 0.013034194314708 w_k1_INTC_s115
                                 + 0.0317378943800473 w_k1_INTC_s116
                                 + 0.0247133287272144 w_k1_INTC_s117
                                 + 0.0250277941060129 w_k1_KO_s000
                                 + 0.0451911339604495 w_k1_KO_s001
                                 + 0.0587311697657286 w_k1_KO_s002
                                 + 0.0283089790582887 w_k1_KO_s003
                                 + 0.0618916494317767 w_k1_KO_s004
                                 + 0.0260124887804959 w_k1_KO_s005
                                 + 0.0272668794189553 w_k1_KO_s006
                                 + 0.0337410231962906 w_k1_KO_s007
                                 + 0.0501256821541494 w_k1_KO_s010
                                 + 0.0290574137339699 w_k1_KO_s011
                                 + 0.0325608703678689 w_k1_KO_s012
                                 + 0.0130808075491201 w_k1_KO_s013
                                 + 0.0162259636600418 w_k1_KO_s014
                                 + 0.0498797074475227 w_k1_KO_s015
                                 + 0.0182893206966734 w_k1_KO_s016
                                 + 0.0162137095916938 w_k1_KO_s017
                                 + 0.051668318638446 w_k1_KO_s100
                                 + 0.0355190938116988 w_k1_KO_s101
                                 + 0.0277212903478175 w_k1_KO_s102
                                 + 0.0303655867851469 w_k1_KO_s103
                                 + 0.0356222486678421 w_k1_KO_s104
                                 + 0.0196739465540257 w_k1_KO_s105
                                 + 0.0424003205078732 w_k1_KO_s106
                                 + 0.0346354845411193 w_k1_KO_s107
                                 + 0.0243999983954245 w_k1_KO_s110
                                 + 0.0297286211879132 w_k1_KO_s111
                                 + 0.0525235014940887 w_k1_KO_s112
                                 + 0.016776279994391 w_k1_KO_s113
                                 + 0.0144086820375095 w_k1_KO_s114
                                 + 0.0143247349934596 w_k1_KO_s115
                                 + 0.0368019939585443 w_k1_KO_s116
                                 + 0.0259327677306214 w_k1_KO_s117
                                 + 0.0267422567657965 w_k1_MAT_s000
                                 + 0.0513600185197961 w_k1_MAT_s001
                                 + 0.0610317428748828 w_k1_MAT_s002
                                 + 0.02518428309732 w_k1_MAT_s003
                                 + 0.0758351928558699 w_k1_MAT_s004
                                 + 0.0224601229886362 w_k1_MAT_s005
                                 + 0.0206905575239339 w_k1_MAT_s006
                                 + 0.0440317902568777 w_k1_MAT_s007
                                 + 0.0461531391795974 w_k1_MAT_s010
                                 + 0.028029832722053 w_k1_MAT_s011
                                 + 0.0331373687942039 w_k1_MAT_s012
                                 + 0.0151804362149049 w_k1_MAT_s013
                                 + 0.0190714713557669 w_k1_MAT_s014
                                 + 0.0520573034658975 w_k1_MAT_s015
                                 + 0.01971518095054 w_k1_MAT_s016
                                 + 0.0165429261479098 w_k1_MAT_s017
                                 + 0.0687946076912935 w_k1_MAT_s100
                                 + 0.0313991319147255 w_k1_MAT_s101
                                 + 0.0230601113807147 w_k1_MAT_s102
                                 + 0.0340118367813708 w_k1_MAT_s103
                                 + 0.032006657193631 w_k1_MAT_s104
                                 + 0.0181432796299832 w_k1_MAT_s105
                                 + 0.0451801028857485 w_k1_MAT_s106
                                 + 0.0387143209389856 w_k1_MAT_s107
                                 + 0.0318043867476578 w_k1_MAT_s110
                                 + 0.0386404875814599 w_k1_MAT_s111
                                 + 0.0580921365103353 w_k1_MAT_s112
                                 + 0.0202313950611249 w_k1_MAT_s113
                                 + 0.015747064755566 w_k1_MAT_s114
                                 + 0.0126436247821931 w_k1_MAT_s115
                                 + 0.0408223187640294 w_k1_MAT_s116
                                 + 0.0260373170184096 w_k1_MAT_s117
                                 + 0.0235613764074449 w_k1_MSFT_s000
                                 + 0.0434432293344557 w_k1_MSFT_s001
                                 + 0.0547911517153039 w_k1_MSFT_s002
                                 + 0.0275074973563641 w_k1_MSFT_s003
                                 + 0.0660334606967154 w_k1_MSFT_s004
                                 + 0.0285864760247069 w_k1_MSFT_s005
                                 + 0.0253430288097929 w_k1_MSFT_s006
                                 + 0.0348164482426815 w_k1_MSFT_s007
                                 + 0.0469149732217391 w_k1_MSFT_s010
                                 + 0.0255234126491154 w_k1_MSFT_s011
                                 + 0.039199090064792 w_k1_MSFT_s012
                                 + 0.0122193511699784 w_k1_MSFT_s013
                                 + 0.0163845981008998 w_k1_MSFT_s014
                                 + 0.0464532193283146 w_k1_MSFT_s015
                                 + 0.0181355662148119 w_k1_MSFT_s016
                                 + 0.0192458014196096 w_k1_MSFT_s017
                                 + 0.0508706355101644 w_k1_MSFT_s100
                                 + 0.0315458766689897 w_k1_MSFT_s101
                                 + 0.0270223642589575 w_k1_MSFT_s102
                                 + 0.0350163201363565 w_k1_MSFT_s103
                                 + 0.0335055981379203 w_k1_MSFT_s104
                                 + 0.0219798981014521 w_k1_MSFT_s105
                                 + 0.0383323013896948 w_k1_MSFT_s106
                                 + 0.0355036854568897 w_k1_MSFT_s107
                                 + 0.0238011960079796 w_k1_MSFT_s110
                                 + 0.0324561822778892 w_k1_MSFT_s111
                                 + 0.0557734296897213 w_k1_MSFT_s112
                                 + 0.0172138605132996 w_k1_MSFT_s113
                                 + 0.0149376670861732 w_k1_MSFT_s114
                                 + 0.0126852821046354 w_k1_MSFT_s115
                                 + 0.0345128760477912 w_k1_MSFT_s116
                                 + 0.0260196588385897 w_k1_MSFT_s117
                                 + 0.020885579831725 w_k1_PG_s000
                                 + 0.0436992567431465 w_k1_PG_s001
                                 + 0.0577428457891472 w_k1_PG_s002
                                 + 0.0265419261610382 w_k1_PG_s003
                                 + 0.0659152778132409 w_k1_PG_s004
                                 + 0.0263650144330433 w_k1_PG_s005
                                 + 0.0246173648963631 w_k1_PG_s006
                                 + 0.0315993209590257 w_k1_PG_s007
                                 + 0.0491072439522053 w_k1_PG_s010
                                 + 0.0264001027139948 w_k1_PG_s011
                                 + 0.0338352453030106 w_k1_PG_s012
                                 + 0.0144790509985097 w_k1_PG_s013
                                 + 0.0169463780257272 w_k1_PG_s014
                                 + 0.0515373432825268 w_k1_PG_s015
                                 + 0.017191345489772 w_k1_PG_s016
                                 + 0.0178947309314649 w_k1_PG_s017
                                 + 0.0544951109415864 w_k1_PG_s100
                                 + 0.0324486689476658 w_k1_PG_s101
                                 + 0.0261674988870862 w_k1_PG_s102
                                 + 0.0314916823023975 w_k1_PG_s103
                                 + 0.0299683595143076 w_k1_PG_s104
                                 + 0.0190703757900034 w_k1_PG_s105
                                 + 0.0427823579855232 w_k1_PG_s106
                                 + 0.0352153216894145 w_k1_PG_s107
                                 + 0.0266437027152817 w_k1_PG_s110
                                 + 0.0322586502531152 w_k1_PG_s111
                                 + 0.0558399696042784 w_k1_PG_s112
                                 + 0.0172622715180391 w_k1_PG_s113
                                 + 0.0138635737044633 w_k1_PG_s114
                                 + 0.0132938039290377 w_k1_PG_s115
                                 + 0.0332516480163591 w_k1_PG_s116
                                 + 0.02410780058367 w_k1_PG_s117 >= 1.041
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
