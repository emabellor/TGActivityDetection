/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.model;

import activitybesa.classdata.CamRelationClass;
import activitybesa.classdata.PersonInfoClass;
import activitybesa.utils.Utils;
import java.awt.Color;
import java.awt.Container;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.MouseInfo;
import java.awt.Point;
import java.awt.event.ComponentEvent;
import java.awt.event.ComponentListener;
import java.awt.event.MouseEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import org.apache.commons.io.FileUtils;

// Normatividad
// Ancho de carriles
// Alcaldia de medellin
// http://servicios.medellin.gov.co/POT/DECRETO_409_2007/ch02s585.html

/**
 *
 * @author mauricio
 */
public class JFrameHandler extends javax.swing.JFrame {
    private List<Image> listImages = new ArrayList<>();
    private List<JLabel> listLabels = new ArrayList<>();
    
    private Date startDate = null;
    private Date endDate = null;
    private Date currentDate = null;
    private boolean mouseSliderPressed = false;
    private List<IFrameEvents> listeners = new ArrayList<>(); 
    private Map<Integer, CamRelationClass> cameraUIMap  = null;
    private int controlPressed = 0;
    
    /**
     * Creates new form JFrameHandler
     */
    public JFrameHandler() {
        initComponents();
       
        int totalImages = 6;
        for (int i = 0; i < totalImages; i++) {
            listImages.add(null);
        }
        
        // Initializing JLabels
        listLabels.add(jLabel1);
        listLabels.add(jLabel2);
        listLabels.add(jLabel3);
        listLabels.add(jLabel4);
        listLabels.add(jLabel5);
        listLabels.add(jLabel6);
    }
    
    public void AddListener(IFrameEvents toAdd) {
        listeners.add(toAdd);
    }

    public void SendEventChangeDate(Date date) {
        listeners.forEach((elem) -> {
            elem.ChangeDate(date);
        });
    }
    
    public void SetDates (Date startDate, Date endDate) {
        this.startDate = startDate;
        this.endDate = endDate;
    
        SimpleDateFormat dateTimeFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        jLabelDateStart.setText("Start: " + dateTimeFormat.format(startDate));
        jLabelDateEnd.setText("End: " + dateTimeFormat.format(endDate));
    }
    
    public void SetCameraUIMap(Map<Integer, CamRelationClass> cameraUIMap) {
        this.cameraUIMap = cameraUIMap;
    }
    
    public void SetSlider(Date currentDate) {
        if (mouseSliderPressed == false) {
            
            double totalMs = Utils.GetDifferenceInMinutesAbs(startDate, endDate);
            double diffMs = Utils.GetDifferenceInMinutesAbs(startDate, currentDate);
            
            int valueSlider = (int)(diffMs * 100 / totalMs);
            videoSlider.setValue(valueSlider);
            
            this.currentDate = currentDate;
        }
    }
    
    public Image GetImage(int idCam) {
        int idCamUI = cameraUIMap.get(idCam).idCamUI;
        return listImages.get(idCamUI);  
    }
    
    public void SetImage(int idCam, Image image) {
        ImageIcon icon = new ImageIcon();
        Image scaledIconImg = Utils.GetScaledImage(image, 320, 240);
        icon.setImage(scaledIconImg);
         
        int idCamUI = cameraUIMap.get(idCam).idCamUI;
        
        // Convert image and save into array!
        listImages.set(idCamUI, image);
        listLabels.get(idCamUI).setIcon(icon); 
    }
    
 
    public void SetPositionPoints(List<PersonInfoClass> listPeople) {
        // Setting position points
        // Creating blank image
        
        BufferedImage blankImg = Utils.CreateBufferedImage(jLabelPos.getWidth(), jLabelPos.getHeight());
        Graphics2D g2D = blankImg.createGraphics();
        g2D.setColor(Color.BLACK);
        
        // Convert points to 2D
        int rectRad = 4;
        
        for (PersonInfoClass person : listPeople) {
            Utils.logger.log(Level.WARNING, "XValue {0}", person.position.x);
            Utils.logger.log(Level.WARNING, "YValue {0}", person.position.y);
            
            // Transform position to image coordinates
            double xFactor = (double)jLabelPos.getWidth() / 1150;
            double yFactor = (double)jLabelPos.getHeight()/ 3000;
              
            double xValue = (200 + person.position.x) * xFactor;
            double yValue = (1500 - person.position.y) * yFactor;
        
            // Local validation before drawing
            if (xValue < 0 || xValue >= jLabelPos.getWidth()) {
                Utils.logger.log(Level.WARNING, "Graphic point outside bounds: x {0}", xValue);
            } else if (yValue < 0 || yValue >= jLabelPos.getHeight()) {
                Utils.logger.log(Level.WARNING, "Graphic point outside bounds: y {0}", yValue);
            } else {
                // Draw point
                g2D.fillRect((int)xValue -rectRad, (int)yValue -rectRad, rectRad * 2, rectRad * 2);
            }
        } 
        
        // Disposing element
        g2D.dispose();
        
        // Convert image and draw
        ImageIcon icon = new ImageIcon();
        icon.setImage(blankImg);
        jLabelPos.setIcon(icon);
        
        // Done!
    }
    
    
    public void SetFactorLabel(int factor) {
        jLabelFactor.setText("Factor: " + factor + "X");
    }
    
    public void SetImage(int idCam, byte[] image) {
        ImageIcon icon = Utils.ByteToImageIcon(image);
        Image unscaled = icon.getImage();
        Image scaledIconImg = Utils.GetScaledImage(icon.getImage(), 320, 240);
        icon.setImage(scaledIconImg);
        
        int idCamUI = cameraUIMap.get(idCam).idCamUI;
        listImages.set(idCamUI, unscaled);
        listLabels.get(idCamUI).setIcon(icon);   
    }
    
    public void UpdateDate(Date date) {
        SimpleDateFormat dateTimeFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");      
        jLabelDate.setText("Date: " + dateTimeFormat.format(date));
        SetSlider(date);
    }
    
    public static class ExitListener extends WindowAdapter {
        @Override
        public void windowClosing(WindowEvent event) {
            System.exit(0);
        }
    }
    

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jPopupMenuMain = new javax.swing.JPopupMenu();
        jMenuItemSaveImage = new javax.swing.JMenuItem();
        jLabel1 = new javax.swing.JLabel();
        jLabel3 = new javax.swing.JLabel();
        jLabel2 = new javax.swing.JLabel();
        jLabel6 = new javax.swing.JLabel();
        jLabel4 = new javax.swing.JLabel();
        jLabel5 = new javax.swing.JLabel();
        jLabelDate = new javax.swing.JLabel();
        videoSlider = new javax.swing.JSlider();
        jLabelDateStart = new javax.swing.JLabel();
        jLabelDateEnd = new javax.swing.JLabel();
        jLabelFactor = new javax.swing.JLabel();
        jButtonSetDate = new javax.swing.JButton();
        jLabelPos = new javax.swing.JLabel();

        jMenuItemSaveImage.setText("Save Image");
        jMenuItemSaveImage.addAncestorListener(new javax.swing.event.AncestorListener() {
            public void ancestorMoved(javax.swing.event.AncestorEvent evt) {
            }
            public void ancestorAdded(javax.swing.event.AncestorEvent evt) {
                jMenuItemSaveImageAncestorAdded(evt);
            }
            public void ancestorRemoved(javax.swing.event.AncestorEvent evt) {
            }
        });
        jMenuItemSaveImage.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                jMenuItemSaveImageMouseClicked(evt);
            }
        });
        jMenuItemSaveImage.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jMenuItemSaveImageActionPerformed(evt);
            }
        });
        jPopupMenuMain.add(jMenuItemSaveImage);

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        jLabel1.setBackground(new java.awt.Color(255, 220, 184));
        jLabel1.setText("Video 1");
        jLabel1.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mousePressed(java.awt.event.MouseEvent evt) {
                jLabel1MousePressed(evt);
            }
        });

        jLabel3.setBackground(new java.awt.Color(255, 220, 184));
        jLabel3.setText("Video 3");
        jLabel3.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mousePressed(java.awt.event.MouseEvent evt) {
                jLabel3MousePressed(evt);
            }
        });

        jLabel2.setBackground(new java.awt.Color(192, 192, 192));
        jLabel2.setText("Video 2");
        jLabel2.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mousePressed(java.awt.event.MouseEvent evt) {
                jLabel2MousePressed(evt);
            }
        });

        jLabel6.setBackground(new java.awt.Color(255, 220, 184));
        jLabel6.setText("Video 6");
        jLabel6.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mousePressed(java.awt.event.MouseEvent evt) {
                jLabel6MousePressed(evt);
            }
        });

        jLabel4.setBackground(new java.awt.Color(255, 220, 184));
        jLabel4.setText("Video 4");
        jLabel4.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mousePressed(java.awt.event.MouseEvent evt) {
                jLabel4MousePressed(evt);
            }
        });

        jLabel5.setBackground(new java.awt.Color(255, 220, 184));
        jLabel5.setText("Video 5");
        jLabel5.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mousePressed(java.awt.event.MouseEvent evt) {
                jLabel5MousePressed(evt);
            }
        });

        jLabelDate.setText("Fecha: 2017-02-02 12:00:00");

        videoSlider.setValue(0);
        videoSlider.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mousePressed(java.awt.event.MouseEvent evt) {
                videoSliderMousePressed(evt);
            }
            public void mouseReleased(java.awt.event.MouseEvent evt) {
                videoSliderMouseReleased(evt);
            }
        });

        jLabelDateStart.setText("Start: 2017-02-02 12:00:00");

        jLabelDateEnd.setText("End: 2017-02-02 12:00:00");

        jLabelFactor.setText("Factor: 1x");

        jButtonSetDate.setText("Set Date");
        jButtonSetDate.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButtonSetDateActionPerformed(evt);
            }
        });

        jLabelPos.setBackground(new java.awt.Color(255, 220, 184));
        jLabelPos.setText("Position");
        jLabelPos.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mousePressed(java.awt.event.MouseEvent evt) {
                jLabelPosMousePressed(evt);
            }
        });

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addGap(10, 10, 10)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(jLabelDateEnd, javax.swing.GroupLayout.PREFERRED_SIZE, 290, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(jLabelDate, javax.swing.GroupLayout.PREFERRED_SIZE, 290, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(jLabelFactor, javax.swing.GroupLayout.PREFERRED_SIZE, 290, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(jLabelDateStart, javax.swing.GroupLayout.PREFERRED_SIZE, 290, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(jButtonSetDate, javax.swing.GroupLayout.PREFERRED_SIZE, 134, javax.swing.GroupLayout.PREFERRED_SIZE)))
                        .addGap(49, 49, 49)
                        .addComponent(videoSlider, javax.swing.GroupLayout.PREFERRED_SIZE, 510, javax.swing.GroupLayout.PREFERRED_SIZE))
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(jLabel1, javax.swing.GroupLayout.PREFERRED_SIZE, 320, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addGap(10, 10, 10)
                        .addComponent(jLabel2, javax.swing.GroupLayout.PREFERRED_SIZE, 320, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addGap(18, 18, 18)
                        .addComponent(jLabel3, javax.swing.GroupLayout.PREFERRED_SIZE, 320, javax.swing.GroupLayout.PREFERRED_SIZE))
                    .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                        .addComponent(jLabelPos, javax.swing.GroupLayout.PREFERRED_SIZE, 990, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addGroup(layout.createSequentialGroup()
                            .addComponent(jLabel4, javax.swing.GroupLayout.PREFERRED_SIZE, 320, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addGap(10, 10, 10)
                            .addComponent(jLabel5, javax.swing.GroupLayout.PREFERRED_SIZE, 320, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addGap(20, 20, 20)
                            .addComponent(jLabel6, javax.swing.GroupLayout.PREFERRED_SIZE, 320, javax.swing.GroupLayout.PREFERRED_SIZE))))
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(videoSlider, javax.swing.GroupLayout.PREFERRED_SIZE, 72, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(jLabelDate)
                        .addGap(12, 12, 12)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                            .addComponent(jLabelDateStart)
                            .addComponent(jButtonSetDate))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addComponent(jLabelDateEnd)))
                .addGap(18, 18, 18)
                .addComponent(jLabelFactor, javax.swing.GroupLayout.PREFERRED_SIZE, 20, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jLabel2, javax.swing.GroupLayout.PREFERRED_SIZE, 240, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel3, javax.swing.GroupLayout.PREFERRED_SIZE, 240, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel1, javax.swing.GroupLayout.PREFERRED_SIZE, 240, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(10, 10, 10)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(jLabel4, javax.swing.GroupLayout.PREFERRED_SIZE, 240, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel5, javax.swing.GroupLayout.PREFERRED_SIZE, 240, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel6, javax.swing.GroupLayout.PREFERRED_SIZE, 240, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addComponent(jLabelPos, javax.swing.GroupLayout.PREFERRED_SIZE, 240, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void videoSliderMousePressed(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_videoSliderMousePressed
        mouseSliderPressed = true;
    }//GEN-LAST:event_videoSliderMousePressed

    private void videoSliderMouseReleased(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_videoSliderMouseReleased
        mouseSliderPressed = false;

        // Calculate date for start and end ms
        Utils.logger.log(Level.FINE, "Slider Value: {0}", videoSlider.getValue());

        double totalMs = Utils.GetDifferenceInMs(startDate, endDate);
        double elapsedMs = totalMs * videoSlider.getValue() / 100;
        Date elapsedDate = Utils.AddDateMs(startDate, (int)elapsedMs);

        // Sending change event
        this.currentDate = elapsedDate;
        SendEventChangeDate(elapsedDate);
    }//GEN-LAST:event_videoSliderMouseReleased

    private void jButtonSetDateActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButtonSetDateActionPerformed
        // Setting dialog using default boxes

        SimpleDateFormat dateTimeFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        String dateHint = dateTimeFormat.format(this.currentDate);
        
        String s = (String)JOptionPane.showInputDialog(
                            this,
                            "Select the date you want to set in the model",
                            "Dialog",
                            JOptionPane.PLAIN_MESSAGE,
                            null,
                            null,
                            dateHint);

        //If a string was returned, say so.
        if ((s != null) && (s.length() > 0)) {
            // Try to convert with formatter 
            try {
                Date dateToSet = dateTimeFormat.parse(s);
                
                double diffMsStart = Utils.GetDifferenceInMinutes(startDate, dateToSet);
                double diffMsEnd = Utils.GetDifferenceInMinutes(dateToSet, endDate);
                
                if (diffMsStart < 0) {
                    JOptionPane.showMessageDialog(this,
                        "StartDate must be lower than DateToSet",
                        "Error",
                        JOptionPane.ERROR_MESSAGE);
                } else if (diffMsEnd < 0) {
                    JOptionPane.showMessageDialog(this,
                        "DateToSet must be lower than EndDate",
                        "Error",
                        JOptionPane.ERROR_MESSAGE);
                } else {
                    // Send event to world
                    SetSlider(dateToSet);
                    SendEventChangeDate(dateToSet); 
                }              
            } catch (ParseException ex) {
                JOptionPane.showMessageDialog(this,
                    "Error parsing date!",
                    "Error",
                    JOptionPane.ERROR_MESSAGE);
            }
        }
    }//GEN-LAST:event_jButtonSetDateActionPerformed

    private void jLabel1MousePressed(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_jLabel1MousePressed
        if (evt.getButton() == MouseEvent.BUTTON3) {
            // Showing element
            controlPressed = 0;
            
            Point pos = MouseInfo.getPointerInfo().getLocation();    
            Point posWindow = this.getLocationOnScreen();
            jPopupMenuMain.show(this, pos.x - posWindow.x, pos.y - posWindow.y);
        }
    }//GEN-LAST:event_jLabel1MousePressed

    private void jLabel2MousePressed(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_jLabel2MousePressed
        if (evt.getButton() == MouseEvent.BUTTON3) {
            // Showing element
            controlPressed = 1;
            
            Point pos = MouseInfo.getPointerInfo().getLocation();    
            Point posWindow = this.getLocationOnScreen();
            jPopupMenuMain.show(this, pos.x - posWindow.x, pos.y - posWindow.y);
        }
    }//GEN-LAST:event_jLabel2MousePressed

    private void jLabel3MousePressed(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_jLabel3MousePressed
        if (evt.getButton() == MouseEvent.BUTTON3) {
            // Showing element
            controlPressed = 2;
            
            Point pos = MouseInfo.getPointerInfo().getLocation();    
            Point posWindow = this.getLocationOnScreen();
            jPopupMenuMain.show(this, pos.x - posWindow.x, pos.y - posWindow.y);
        }
    }//GEN-LAST:event_jLabel3MousePressed

    private void jLabel4MousePressed(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_jLabel4MousePressed
        if (evt.getButton() == MouseEvent.BUTTON3) {
            // Showing element
            controlPressed = 3;
            
            Point pos = MouseInfo.getPointerInfo().getLocation();    
            Point posWindow = this.getLocationOnScreen();
            jPopupMenuMain.show(this, pos.x - posWindow.x, pos.y - posWindow.y);
        }
    }//GEN-LAST:event_jLabel4MousePressed

    private void jLabel5MousePressed(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_jLabel5MousePressed
        if (evt.getButton() == MouseEvent.BUTTON3) {
            // Showing element
            controlPressed = 4;
            
            Point pos = MouseInfo.getPointerInfo().getLocation();    
            Point posWindow = this.getLocationOnScreen();
            jPopupMenuMain.show(this, pos.x - posWindow.x, pos.y - posWindow.y);
        }
    }//GEN-LAST:event_jLabel5MousePressed

    private void jLabel6MousePressed(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_jLabel6MousePressed
        if (evt.getButton() == MouseEvent.BUTTON3) {
            // Showing element
            controlPressed = 5;
            
            Point pos = MouseInfo.getPointerInfo().getLocation();    
            Point posWindow = this.getLocationOnScreen();
            jPopupMenuMain.show(this, pos.x - posWindow.x, pos.y - posWindow.y);
        }
    }//GEN-LAST:event_jLabel6MousePressed

    private void jMenuItemSaveImageMouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_jMenuItemSaveImageMouseClicked
        
    }//GEN-LAST:event_jMenuItemSaveImageMouseClicked

    private void jMenuItemSaveImageAncestorAdded(javax.swing.event.AncestorEvent evt) {//GEN-FIRST:event_jMenuItemSaveImageAncestorAdded
       
    }//GEN-LAST:event_jMenuItemSaveImageAncestorAdded

    private void jMenuItemSaveImageActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jMenuItemSaveImageActionPerformed
        try {
            Image imageRef = listImages.get(controlPressed);           
            byte[] image = Utils.ImageToByteArray(imageRef);
        
            if (image == null) {
                Utils.ShowErrorBox("No existe imagen para el control asociado");
            } else {
                // Loading automatically
                Utils.CheckDirectory(Utils.CALIB_IMAGE_FOLDER);

                int idCam = GetIdCamFromCamUI(controlPressed);
                Path pathFile = Paths.get(Utils.CALIB_IMAGE_FOLDER, String.valueOf(idCam) + ".jpg");

                FileUtils.writeByteArrayToFile(pathFile.toFile(), image);                
                Utils.ShowMessageBox("Image saved in " + pathFile.toString());
            }  
        } catch (IOException ex) {
            Utils.ShowErrorBox("Exception saving image file: " + ex.toString());
        }
    }//GEN-LAST:event_jMenuItemSaveImageActionPerformed

    private void jLabelPosMousePressed(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_jLabelPosMousePressed
        // TODO add your handling code here:
    }//GEN-LAST:event_jLabelPosMousePressed

    private int GetIdCamFromCamUI(int idCamUI) {
        boolean found = false;
        int idCam = 0;
        
        for (Map.Entry<Integer, CamRelationClass> entry : cameraUIMap.entrySet()) {
            if (entry.getValue().idCamUI == idCamUI) {
                idCam = entry.getKey();
                found = true;
                break;
            }
        }
        
        if (found == false) {
            throw new IllegalArgumentException("There was no map for idCamUI: " + idCamUI);
        } else {
            return idCam;
        }
    }
    

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton jButtonSetDate;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JLabel jLabel3;
    private javax.swing.JLabel jLabel4;
    private javax.swing.JLabel jLabel5;
    private javax.swing.JLabel jLabel6;
    private javax.swing.JLabel jLabelDate;
    private javax.swing.JLabel jLabelDateEnd;
    private javax.swing.JLabel jLabelDateStart;
    private javax.swing.JLabel jLabelFactor;
    private javax.swing.JLabel jLabelPos;
    private javax.swing.JMenuItem jMenuItemSaveImage;
    private javax.swing.JPopupMenu jPopupMenuMain;
    private javax.swing.JSlider videoSlider;
    // End of variables declaration//GEN-END:variables
}
