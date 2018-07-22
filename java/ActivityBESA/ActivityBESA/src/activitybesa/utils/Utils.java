/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.utils;

import BESA.ExceptionBESA;
import BESA.Kernell.Agent.AgentBESA;
import BESA.Kernell.Agent.Event.DataBESA;
import BESA.Kernell.Agent.Event.EventBESA;
import BESA.Kernell.System.Directory.AgHandlerBESA;
import BESA.Log.ReportBESA;
import activitybesa.ClassMain;
import activitybesa.classdata.ResultsClass;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;
import java.util.List;
import java.util.UUID;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JOptionPane;

/**
 *
 * @author mauricio
 */
public class Utils {
    // Logging constants
    public static final Logger logger = Logger.getLogger(ClassMain.class.getName()); 
   
    static {
        logger.setUseParentHandlers(false);

        MyFormatter formatter = new MyFormatter();
        ConsoleHandler handler = new ConsoleHandler();
        handler.setFormatter(formatter);
        handler.setLevel(Level.FINEST);
        logger.addHandler(handler);
        logger.setLevel(Level.FINEST);
    }
            
    // Global variable naming!
    public static double MIN_COLOR_DISTANCE = 45;
    public static double MIN_POSE_SCORE = 0.01;
    public static int GAME_PERIODIC_TIME = 500;
    public static int GAME_PERIODIC_DELAY_TIME = 100;
    public static int MIN_DISTANCE_TRACKING = 320;
    public static String CALIB_BASE_FOLDER = "/home/mauricio/Oviedo/CameraCalibration";
    public static String CALIB_IMAGE_FOLDER = "/home/mauricio/Oviedo/CameraImages";
    public static Gson gson = new GsonBuilder().disableHtmlEscaping().create();
       
    // Settings
    public static final boolean allowNoTraining = false;
    
    public static ImageIcon ByteToImageIcon(byte[] image) {
        // Image must be in png or jpeg
        return new ImageIcon(image);
    }
    
    public static int GetIdCamFromAlias(String alias) {
        // Camera names are in the form CAM_0
        String[] params = alias.split("_");
        return Integer.parseInt(params[1]);
    }
    
    public static void SendEventBesa(AgentBESA sender, String alias, Class guard, DataBESA data) {
        EventBESA event = new EventBESA(guard.getName(), data);
        AgHandlerBESA ah;
        try {
            ah = sender.getAdmLocal().getHandlerByAlias(alias);
            ah.sendEvent(event);
        } catch (ExceptionBESA e) {
            ReportBESA.error(e);
        }
    }
    
    public static void SendEventBesaWorld(AgentBESA sender, Class guard, DataBESA data) {
        SendEventBesa(sender, "WORLD", guard, data);
    }
    
    public static BufferedImage GetScaledImage(Image srcImg, int w, int h){
        BufferedImage resizedImg = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2 = resizedImg.createGraphics();

        g2.setRenderingHint(RenderingHints
                .KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g2.drawImage(srcImg, 0, 0, w, h, null);
        g2.dispose();

        return resizedImg;
    }
    
    public static Image DrawLine(Image srcImg, Point pt1, Point pt2, Color color) {
        // Create a buffered image with transparency
        BufferedImage bimage = new BufferedImage(srcImg.getWidth(null), srcImg.getHeight(null),
                BufferedImage.TYPE_INT_ARGB);

        // Draw the image on to the buffered image
        Graphics2D bGr = bimage.createGraphics();
        bGr.drawImage(srcImg, 0, 0, null);
          
        // Draw the line into image
        bGr.setColor(color);
        bGr.setStroke(new BasicStroke(4));
        
        bGr.drawLine(pt1.x, pt1.y, pt2.x, pt2.y);
        bGr.dispose();

        // Return the buffered image
        return bimage;
    }
    
    public static void DrawLinesGraphics(Graphics2D bGr, List<LineInfo> listPoints) {          
        bGr.setStroke(new BasicStroke(4));
        
        for (int i = 0; i < listPoints.size(); i++) {
            LineInfo info = listPoints.get(i);
            
            bGr.setColor(info.color);
            bGr.drawLine(info.pt1.x, info.pt1.y, info.pt2.x, info.pt2.y);
        }
    }
    
    public static Image DrawLines(Image srcImg, List<LineInfo> listPoints) {
        // Create a buffered image with transparency
        BufferedImage bimage = new BufferedImage(srcImg.getWidth(null), srcImg.getHeight(null),
                BufferedImage.TYPE_INT_ARGB);

        // Draw the image on to the buffered image
        Graphics2D bGr = bimage.createGraphics();
        bGr.drawImage(srcImg, 0, 0, null);
        DrawLinesGraphics(bGr, listPoints);
        bGr.dispose();

        // Return the buffered image
        return bimage;
    }
   
    public static void DrawRectangleGraphics(Graphics2D bGr, Rectangle rect, String caption) {        
        // Less basic stroke for image
        bGr.setStroke(new BasicStroke(5)); 
        bGr.setColor(new Color(255, 255, 255));
        bGr.drawRect(rect.x, rect.y, rect.width, rect.height);
       
        if (caption != null) {
            bGr.setFont(new Font("Arial Black", Font.BOLD, 20));
            bGr.drawString(caption, rect.x, rect.y + 20);
        }
    }
    
    public static Image DrawRectangle(Image srcImg, Rectangle rect, String caption) {
        // Same strategy - Points of elements
        BufferedImage bimage = new BufferedImage(srcImg.getWidth(null), srcImg.getHeight(null),
                                    BufferedImage.TYPE_INT_ARGB);

        // Draw the image on to the buffered image
        Graphics2D bGr = bimage.createGraphics();
        bGr.drawImage(srcImg, 0, 0, null);
        DrawRectangleGraphics(bGr, rect, caption);

        // Freeing resources - Must have this implemented
        bGr.dispose();
        
        // Return the buffered image
        return bimage;
    }
    
    public static double GetEuclideanDistance(Point2f p1, Point2f p2) {
        double result = Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2);
        result = Math.sqrt(result);
        return result;
    }
    
    public static double GetHorizontalDistance(Point2f p1, Point2f p2) {
        double result = p1.x - p2.x;
        return result;
    }
    
    public static int BytesToInt(byte[] bytes) {      
        ByteBuffer wrapped = ByteBuffer.wrap(bytes); // big-endian by default
        wrapped.order(ByteOrder.LITTLE_ENDIAN);
        int num = wrapped.getInt(); // 1

        return num;
    }
    
    public static Point GetCenterMassFromResults(List<ResultsClass> results) {
        // Estimate mean over elements
        // Assume mean over elements - Iteration
       
        int x = 0;
        int y = 0;
        for (int i = 0; i < results.size(); i++) {
            x += results.get(i).x;
            y += results.get(i).y;
        }
        
        x = x / results.size();
        y = y / results.size();
        
        return new Point(x, y);
    }
    
    public static String GenerateGUID() {
        return UUID.randomUUID().toString();
    }
    
    public static Rectangle CalculateBoundingRect(List<ResultsClass> results) {
        // TODO - Review this
        // Only from 0 to 14
        // Ignore results under min score
        
        // Assume first element in list - Head
        double x_min = -1;
        double y_min = -1;
        
        double x_max = -1;
        double y_max = -1;
        
        for (int i = 0; i < 14; i++) {
            ResultsClass elem = results.get(i);
            
            if (elem.score < MIN_POSE_SCORE) {
                // Ignore!
            } else {
                if (elem.x < x_min || x_min == -1) {
                    x_min = elem.x;
                }

                if (elem.y < y_min || y_min == -1) {
                    y_min = elem.y;
                }

                if (elem.x > x_max || x_max == -1) {
                    x_max = elem.x;
                }

                if (elem.y > y_max || y_max == -1) {
                    y_max = elem.y;
                }
            }  
        }
       
        int x = (int)x_min;
        int y = (int)y_min;
        int width = (int)(x_max - x_min);
        int height = (int)(y_max - y_min);
        
        return new Rectangle(x, y, width, height);
    }
    
    public static BufferedImage ByteArrayToImage(byte[] imageBin) throws IOException {     
        BufferedImage img = ImageIO.read(new ByteArrayInputStream(imageBin));
        return img; 
    }
        
    public static String GetFileExtension(String fileName) {
        String[] elems = fileName.split("\\.");
        return elems[elems.length -1]; 
    }
    
    public static void KillProgram() {
        System.exit(1);
    }
    
    public static Date TicksToDate(long ticks) {
        //long ticks = 635556672000000000L; 

        long ticksMilli = ticks / (10 * 1000);
        long epochMilli = 62135596800000L;
        
        //new date is ticks, converted to microtime, minus difference from epoch microtime
        Date tickDate = new Date(ticksMilli - epochMilli);
    
        return tickDate;
    }
    
    public static long DateToTicks(Date date) {
        long epochMicrotimeDiff = 62135596800000L;
        
        long ticksMicroTime =  date.getTime() + epochMicrotimeDiff;
        return ticksMicroTime * (10 * 1000); // NanoTime!
    }

    public static Date TicksToDate(byte[] ticksBin) {
        // Annotation - Must be little endian!
        ByteBuffer buffer = ByteBuffer.allocate(Long.BYTES);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        buffer.put(ticksBin, 0, ticksBin.length);
       
        long ticks = buffer.getLong(0);
        return TicksToDate(ticks);
    }
    
    public static String GetVideoPath(String baseFolder, long ticks, int idCam) {
        Date date = TicksToDate(ticks);
        return GetVideoPath(baseFolder, date, idCam);
    }
    
    public static String GetVideoPath(String baseFolder, Date dateFrame, int idCam) {
        // Erase 15 minute gap
        Calendar calendar = Calendar.getInstance();
        calendar.setTime(dateFrame);
   
        int totalMinutes = calendar.get(Calendar.MINUTE);
        int remaining = totalMinutes % 15;
        calendar.set(Calendar.SECOND, 0);
        calendar.set(Calendar.MINUTE, totalMinutes - remaining); // Method to visualize videos - Chunks of 15 minutes
        
        Date newDate = new Date(calendar.getTimeInMillis());

        SimpleDateFormat dayFormat = new SimpleDateFormat("yyyy-MM-dd");
        
        SimpleDateFormat hourFormat = new SimpleDateFormat("HH-mm-ss");
        
        String folderName = dayFormat.format(newDate);        
        String fileName = hourFormat.format(newDate) + ".mjpegx";
             
        Path pathFile = Paths.get(baseFolder, folderName, String.valueOf(idCam), fileName);
        
        String pathFileString = pathFile.toString();     
        return pathFileString; 
    }
    
    public static Date AddDateMs(Date dateInit, int ms){
        Calendar calendar = Calendar.getInstance();
        calendar.setTime(dateInit);
        calendar.add(Calendar.MILLISECOND, ms);
        return new Date(calendar.getTimeInMillis());
    }
    
    public static Date AddDateHours(Date dateInit, int hours){
        Calendar calendar = Calendar.getInstance();
        calendar.setTime(dateInit);
        calendar.add(Calendar.HOUR, hours);
        return new Date(calendar.getTimeInMillis());
    }
    
    public static boolean FileExists(String filePath) {
        File f = new File(filePath);
        if (f.exists() && !f.isDirectory()) { 
            return true;
        } else {
           return false; 
        }
    }
    
    public static boolean FileExists(Path filePath) {
        File f = filePath.toFile();
        if (f.exists() && !f.isDirectory()) { 
            return true;
        } else {
           return false; 
        }
    }
    
    public static void CheckDirectory(String pathDir) {
        File f = new File(pathDir);
        if (f.exists() == false) { 
            f.mkdir();
        } 
    } 
    
    public static double GetDifferenceInMs(Date startDate, Date endDate) {
        long diffInMillies = endDate.getTime() - startDate.getTime();
        return diffInMillies;
    }
    
    public static double GetDifferenceInMinutes(Date startDate, Date endDate) {
        long diffInMillies = endDate.getTime() - startDate.getTime();

        double diffInMinutes = (double)diffInMillies / (1000 * 60);
        return diffInMinutes;
    }
    
    public static double GetDifferenceInMinutesAbs(Date startDate, Date endDate) {
        long diffInMillies = endDate.getTime() - startDate.getTime();

        double diffInMinutes = (double)diffInMillies / (1000 * 60);
        return Math.abs(diffInMinutes); 
    }
    
    public static Image LoadImageFromFile(String file) {
        BufferedImage img = null;
        try {
            img = ImageIO.read(new File(file));
        } catch (IOException e) {
            Utils.logger.log(Level.SEVERE, "Exception: {0} : " + file, e.toString());
            Utils.KillProgram();
        }
        
        return img;
    }
    
    public static Date DateFromString(String dateString) {
        Date date = null;
        
        try {
            DateFormat format = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
            date = format.parse(dateString);
            
            return date;
        } catch (ParseException ex) {
            Utils.logger.log(Level.SEVERE, "Error parsing string date: {0}", dateString);
            Utils.KillProgram();
        } 
        
        return date;
    }

    public static String ReadAllText(String fileName) {
        String text = "";
        try {
            text = new String(Files.readAllBytes(Paths.get(fileName)), StandardCharsets.UTF_8);
        } catch(IOException ex) {
            System.out.println("Exception thrown: " + ex.toString());
            KillProgram();
        }
        
        return text;
    }
    
    public static void ShowMessageBox(String message) {
        JOptionPane.showMessageDialog(null, message, "Mensaje", JOptionPane.INFORMATION_MESSAGE);
    }
    
    public static void ShowErrorBox(String message) {
        JOptionPane.showMessageDialog(null, message, "Mensaje", JOptionPane.ERROR_MESSAGE);
    }
    
    public static void SaveImage(BufferedImage image, String pathImage) {
        try {
            File outputfile = new File(pathImage);
            ImageIO.write(image, "jpg", outputfile);
        } catch (IOException ex) {
            Utils.ShowErrorBox("Exception loading image: " + ex.toString());
        }
    }
    
    public static byte[] ImageToByteArray(Image image) {
        byte[] returnValue;
        
        try {
            BufferedImage bufImage = ToBufferedImage(image);
            
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ImageIO.write(bufImage, "jpg", baos);
            returnValue = baos.toByteArray();
        } catch(IOException ex) {
            Utils.ShowErrorBox("Exception in the app: " + ex.toString());
            returnValue = new byte[0];
        }
        
        return returnValue;
    }    
    
    public static BufferedImage ToBufferedImage(Image img) {
        // Create a buffered image with transparency
        BufferedImage bimage = new BufferedImage(img.getWidth(null), img.getHeight(null), BufferedImage.TYPE_3BYTE_BGR);

        // Draw the image on to the buffered image
        Graphics2D bGr = bimage.createGraphics();
        bGr.drawImage(img, 0, 0, null);
        bGr.dispose();

        // Return the buffered image
        return bimage;
    }
    
    public static BufferedImage CreateBufferedImage(int width, int height) {
        // Creating image
        BufferedImage bimage = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
        
        // Fill white
        Graphics2D bGr = bimage.createGraphics();
        bGr.setColor(Color.white);
        bGr.fillRect(0, 0, width, height);
        bGr.dispose();
        
        return bimage;
    }
}
